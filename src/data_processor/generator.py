import numpy as np
import random
import csv
import os
import json
from .. import config

class DataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_synthetic(self, save_dir: str):
        """
        Generates synthetic instance data based on configuration parameters
        and saves it to standard CSV files in the specified directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        if config.NUM_SWAPPING < config.NUM_BASES:
            raise ValueError(
                f"NUM_SWAPPING ({config.NUM_SWAPPING}) must be >= NUM_BASES ({config.NUM_BASES}) to ensure each base has a colocated swap station."
            )
        
        locations_path = os.path.join(save_dir, "locations.csv")
        orders_path = os.path.join(save_dir, "orders.csv")
        
        locations = []
        base_locations = []
        for i in range(1, config.NUM_BASES + 1):
            x_val = np.random.uniform(0, config.AREA_SIZE)
            y_val = np.random.uniform(0, config.AREA_SIZE)
            base_locations.append((x_val, y_val))
            locations.append({"index": f"L{i}", "type": "L", "x": x_val, "y": y_val})
        for i in range(1, config.NUM_HUBS + 1):
            locations.append({"index": f"H{i}", "type": "H", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
        # Ensure every base has at least one colocated swap station.
        for i, (x_val, y_val) in enumerate(base_locations, 1):
            locations.append({"index": f"E{i}", "type": "E", "x": x_val, "y": y_val})
        for i in range(config.NUM_BASES + 1, config.NUM_SWAPPING + 1):
            locations.append({"index": f"E{i}", "type": "E", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
            
        with open(locations_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["index", "type", "x", "y"])
            writer.writeheader()
            writer.writerows(locations)
            
        orders = []
        hubs = [loc["index"] for loc in locations if loc["type"] == "H"]
        for p in hubs:
            for d in hubs:
                if p != d and np.random.random() < config.DEMAND_PROBABILITY:
                    demand = np.random.uniform(config.DEMAND_MIN, config.DEMAND_MAX)
                    orders.append({"start_node": p, "end_node": d, "demand": demand})
                    
        with open(orders_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["start_node", "end_node", "demand"])
            writer.writeheader()
            writer.writerows(orders)
        print(f"Synthetic instance generated and saved to: {save_dir}")

    def generate_meituan_hourly(self, raw_data_path: str, save_dir: str,
                                num_hubs: int = 100,
                                num_bases: int = 25, num_swapping: int = 50,
                                date_filter: int = 20221017,
                                map_half_size_km: float = None,
                                candidate_date_filter=None,
                                demand_date_filter=None,
                                candidate_hour_start: int = None,
                                candidate_hour_end: int = None,
                                demand_hour_start: int = None,
                                demand_hour_end: int = None,
                                colocate_base_swaps: bool = None):
        import pandas as pd
        from sklearn.cluster import KMeans
        os.makedirs(save_dir, exist_ok=True)
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

        if colocate_base_swaps is None:
            colocate_base_swaps = bool(getattr(config, "MEITUAN_COLOCATE_BASE_SWAPS", True))

        if colocate_base_swaps and num_swapping < num_bases:
            raise ValueError(
                f"num_swapping ({num_swapping}) must be >= num_bases ({num_bases}) when colocate_base_swaps=True."
            )

        if map_half_size_km is None:
            map_half_size_km = config.MEITUAN_MAP_HALF_SIZE_KM

        def _date_list(value, fallback):
            if value is None:
                value = fallback
            if isinstance(value, (list, tuple, set)):
                values = [int(v) for v in value]
            else:
                values = [int(value)]
            if not values:
                if isinstance(fallback, (list, tuple, set)):
                    values = [int(v) for v in fallback]
                else:
                    values = [int(fallback)]
            return sorted(set(values))

        def _get_config_dates(name, fallback):
            value = getattr(config, name, None)
            return _date_list(value, fallback) if value else _date_list(fallback, fallback)

        candidate_dates = _date_list(
            candidate_date_filter,
            _get_config_dates("MEITUAN_CANDIDATE_DATES", date_filter),
        )
        demand_dates = _date_list(
            demand_date_filter,
            _get_config_dates("MEITUAN_DEMAND_DATES", date_filter),
        )

        if candidate_hour_start is None:
            candidate_hour_start = int(getattr(config, "MEITUAN_CANDIDATE_HOUR_START", config.MEITUAN_WORK_HOUR_START))
        if candidate_hour_end is None:
            candidate_hour_end = int(getattr(config, "MEITUAN_CANDIDATE_HOUR_END", config.MEITUAN_WORK_HOUR_END + 1))
        if demand_hour_start is None:
            demand_hour_start = int(getattr(config, "MEITUAN_DEMAND_HOUR", 12))
        if demand_hour_end is None:
            demand_hour_end = demand_hour_start + 1

        def _hour_duration(start_hour, end_hour):
            duration = (int(end_hour) - int(start_hour)) % 24
            if duration <= 0:
                raise ValueError("Hour windows must have positive duration under [start, end) semantics.")
            return duration

        candidate_hour_count = _hour_duration(candidate_hour_start, candidate_hour_end)
        demand_hour_count = _hour_duration(demand_hour_start, demand_hour_end)

        def _filter_period(frame, dates, start_hour, end_hour):
            subset = frame[frame["dt"].isin(dates)].copy()
            if subset.empty:
                return subset

            order_ts = pd.to_datetime(subset["platform_order_time"], unit="s", utc=True).dt.tz_convert(config.MEITUAN_TIMEZONE)
            subset["order_hour"] = order_ts.dt.hour

            if start_hour < end_hour:
                mask = (subset["order_hour"] >= start_hour) & (subset["order_hour"] < end_hour)
            else:
                mask = (subset["order_hour"] >= start_hour) | (subset["order_hour"] < end_hour)
            return subset[mask].copy()

        def _project(frame, lon_center, lat_center):
            projected = frame.copy()
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * np.cos(lat_center * np.pi / 180)
            projected["sender_x_km"] = (projected["sender_lng"] / 1e6 - lon_center) * km_per_deg_lon
            projected["sender_y_km"] = (projected["sender_lat"] / 1e6 - lat_center) * km_per_deg_lat
            projected["recipient_x_km"] = (projected["recipient_lng"] / 1e6 - lon_center) * km_per_deg_lon
            projected["recipient_y_km"] = (projected["recipient_lat"] / 1e6 - lat_center) * km_per_deg_lat
            return projected

        def _clip_area(frame):
            return frame[
                frame["sender_x_km"].between(-map_half_size_km, map_half_size_km)
                & frame["sender_y_km"].between(-map_half_size_km, map_half_size_km)
                & frame["recipient_x_km"].between(-map_half_size_km, map_half_size_km)
                & frame["recipient_y_km"].between(-map_half_size_km, map_half_size_km)
            ].copy()

        def _weighted_kmeans(points, weights, n_clusters, label):
            if n_clusters <= 0:
                return []
            if len(points) < n_clusters:
                raise ValueError(
                    f"Cannot generate {n_clusters} {label} candidates from only {len(points)} seed points."
                )
            if n_clusters == 1:
                weights_arr = np.asarray(weights, dtype=float)
                weights_arr = weights_arr / max(weights_arr.sum(), 1.0)
                center = np.average(np.asarray(points, dtype=float), axis=0, weights=weights_arr)
                return [tuple(float(v) for v in center)]

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(np.asarray(points, dtype=float), sample_weight=np.asarray(weights, dtype=float))
            ranked_labels = (
                pd.DataFrame({"label": labels, "weight": weights})
                .groupby("label")["weight"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            return [
                (float(km.cluster_centers_[cluster][0]), float(km.cluster_centers_[cluster][1]))
                for cluster in ranked_labels
            ]

        print(f"Generating Meituan instance: H={num_hubs}, L={num_bases}, E={num_swapping}, hour={demand_hour_start}")
        df = pd.read_csv(raw_data_path)

        candidate_df = _filter_period(df, candidate_dates, candidate_hour_start, candidate_hour_end)
        if candidate_df.empty:
            raise ValueError(
                f"No candidate orders found for dates={candidate_dates}, hours=[{candidate_hour_start}, {candidate_hour_end})."
            )

        demand_df = _filter_period(df, demand_dates, demand_hour_start, demand_hour_end)
        if demand_df.empty:
            raise ValueError(
                f"No demand orders found for dates={demand_dates}, hours=[{demand_hour_start}, {demand_hour_end})."
            )

        # Use the candidate window to define a stable local coordinate system.
        top_poi = candidate_df["poi_id"].value_counts().idxmax()
        center_node = candidate_df[candidate_df["poi_id"] == top_poi].iloc[0]
        lon_center = center_node["sender_lng"] / 1e6
        lat_center = center_node["sender_lat"] / 1e6

        candidate_area = _clip_area(_project(candidate_df, lon_center, lat_center))
        demand_area = _clip_area(_project(demand_df, lon_center, lat_center))

        if candidate_area.empty:
            raise ValueError(
                f"No candidate orders in map window half-size {map_half_size_km}km."
            )
        if demand_area.empty:
            raise ValueError(
                f"No demand orders in map window half-size {map_half_size_km}km."
            )

        p_stats = candidate_area.groupby(
            ["sender_lng", "sender_lat", "sender_x_km", "sender_y_km"]
        ).size().reset_index(name="activity_count")
        d_stats = candidate_area.groupby(
            ["recipient_lng", "recipient_lat", "recipient_x_km", "recipient_y_km"]
        ).size().reset_index(name="activity_count")
        p_stats.columns = ["lng", "lat", "x_km", "y_km", "activity_count"]
        d_stats.columns = ["lng", "lat", "x_km", "y_km", "activity_count"]

        node_stats = pd.concat([p_stats, d_stats]).groupby(
            ["lng", "lat", "x_km", "y_km"]
        )["activity_count"].sum().reset_index()

        # Hub candidates are generated from the candidate window only.
        n_h = max(1, min(num_hubs, len(node_stats)))
        h_xy = node_stats[["x_km", "y_km"]].to_numpy()
        h_w = node_stats["activity_count"].to_numpy()
        h_kmeans = KMeans(n_clusters=n_h, random_state=42, n_init="auto")
        node_stats["cluster"] = h_kmeans.fit_predict(h_xy, sample_weight=h_w)

        h_cluster_rank = (
            node_stats.groupby("cluster")["activity_count"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        h_cluster_to_id = {c: f"H{i + 1}" for i, c in enumerate(h_cluster_rank)}
        h_activity = node_stats.groupby("cluster")["activity_count"].sum().to_dict()

        locations = []
        h_id_to_xy = {}
        h_id_to_activity = {}
        for cluster_id in h_cluster_rank:
            h_id = h_cluster_to_id[cluster_id]
            x_val = float(h_kmeans.cluster_centers_[cluster_id][0])
            y_val = float(h_kmeans.cluster_centers_[cluster_id][1])
            locations.append({"index": h_id, "type": "H", "x": x_val, "y": y_val})
            h_id_to_xy[h_id] = (x_val, y_val)
            h_id_to_activity[h_id] = float(h_activity[cluster_id])

        hub_ids = list(h_id_to_xy.keys())
        hub_points = [h_id_to_xy[h] for h in hub_ids]
        hub_weights = [h_id_to_activity[h] for h in hub_ids]

        base_locations = _weighted_kmeans(hub_points, hub_weights, num_bases, "base")
        for i, (x_val, y_val) in enumerate(base_locations, 1):
            locations.append({"index": f"L{i}", "type": "L", "x": x_val, "y": y_val})

        center_arr = np.asarray([h_id_to_xy[h] for h in hub_ids], dtype=float)

        def _assign_nearest(frame, x_col, y_col):
            pts = frame[[x_col, y_col]].to_numpy(dtype=float)
            dist_sq = ((pts[:, None, :] - center_arr[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(dist_sq, axis=1)
            return [hub_ids[idx] for idx in nearest]

        # Demand for the requested operating period. This is averaged over all
        # selected dates and hours, so a one-hour window over N dates is the
        # average corresponding-hour demand.
        demand_area = demand_area.copy()
        demand_area["p_id"] = _assign_nearest(demand_area, "sender_x_km", "sender_y_km")
        demand_area["d_id"] = _assign_nearest(demand_area, "recipient_x_km", "recipient_y_km")
        demand_area = demand_area[demand_area["p_id"] != demand_area["d_id"]].copy()
        pair_orders = demand_area.groupby(["p_id", "d_id"]).size().reset_index(name="order_count")
        demand_denominator = float(len(demand_dates) * demand_hour_count)
        pair_orders["demand"] = pair_orders["order_count"] / demand_denominator
        pair_orders = pair_orders.rename(columns={"p_id": "start_node", "d_id": "end_node"})
        orders_out = pair_orders[["start_node", "end_node", "demand"]].sort_values(
            ["start_node", "end_node"]
        )

        # Candidate OD midpoints help place non-colocated swap candidates along
        # demand corridors without using random perturbations.
        candidate_mapped = candidate_area.copy()
        candidate_mapped["p_id"] = _assign_nearest(candidate_mapped, "sender_x_km", "sender_y_km")
        candidate_mapped["d_id"] = _assign_nearest(candidate_mapped, "recipient_x_km", "recipient_y_km")
        candidate_mapped = candidate_mapped[candidate_mapped["p_id"] != candidate_mapped["d_id"]]
        candidate_pairs = candidate_mapped.groupby(["p_id", "d_id"]).size().reset_index(name="count")

        swap_seed_points = list(hub_points)
        swap_seed_weights = list(hub_weights)
        for row in candidate_pairs.itertuples(index=False):
            p_xy = h_id_to_xy[row.p_id]
            d_xy = h_id_to_xy[row.d_id]
            swap_seed_points.append(((p_xy[0] + d_xy[0]) / 2.0, (p_xy[1] + d_xy[1]) / 2.0))
            swap_seed_weights.append(float(row.count))

        e_locations = []
        if colocate_base_swaps:
            e_locations.extend(base_locations[:num_bases])

        extra_swaps = num_swapping - len(e_locations)
        e_locations.extend(_weighted_kmeans(swap_seed_points, swap_seed_weights, extra_swaps, "swap"))

        for i, (x_val, y_val) in enumerate(e_locations, 1):
            locations.append({"index": f"E{i}", "type": "E", "x": x_val, "y": y_val})

        pd.DataFrame(locations).to_csv(os.path.join(save_dir, "locations.csv"), index=False)

        # Backward-compatible single-period file plus the future multi-period layout.
        orders_out.to_csv(os.path.join(save_dir, "orders.csv"), index=False)
        orders_dir = os.path.join(save_dir, "orders")
        os.makedirs(orders_dir, exist_ok=True)
        period_id = f"h{demand_hour_start:02d}" if demand_hour_count == 1 else f"h{demand_hour_start:02d}_{demand_hour_count}h"
        period_orders_file = f"{period_id}.csv"
        orders_out.to_csv(os.path.join(orders_dir, period_orders_file), index=False)

        pd.DataFrame([
            {
                "period_id": period_id,
                "orders_file": os.path.join("orders", period_orders_file),
                "start_hour": demand_hour_start,
                "end_hour": demand_hour_end,
                "hour_semantics": "[start, end)",
                "dates": ";".join(str(d) for d in demand_dates),
                "num_dates": len(demand_dates),
                "duration_hours": demand_hour_count,
                "raw_orders": len(demand_df),
                "in_area_orders": len(demand_area),
                "retained_od_pairs": len(orders_out),
                "total_hourly_demand": float(orders_out["demand"].sum()),
            }
        ]).to_csv(os.path.join(save_dir, "periods.csv"), index=False)

        print(
            f"Generated {save_dir}: H={len(h_id_to_xy)}, L={num_bases}, E={num_swapping}, "
            f"period={period_id}, OD={len(orders_out)}, demand={orders_out['demand'].sum():.4f}/h"
        )

    def generate_meituan_hierarchical_series(
        self,
        raw_data_path: str,
        save_root: str,
        hub_counts=None,
        num_bases=20,
        num_swapping=20,
        date_filter: int = 20221017,
        map_half_size_km: float = None,
        candidate_date_filter=None,
        demand_date_filter=None,
        candidate_hour_start: int = None,
        candidate_hour_end: int = None,
        demand_hour_start: int = None,
        demand_hour_end: int = None,
        colocate_base_swaps: bool = None,
        hierarchy_activity_weight: float = 0.35,
        hierarchy_balance_weight: float = 0.75,
        hierarchy_overload_tolerance: float = 1.35,
    ):
        """
        Generate a nested Meituan instance series with a shared parent H set.

        The largest H count is generated from raw data once. Smaller instances
        are obtained by weighted agglomerative merges over those parent hubs,
        and every retained hub is an original parent H candidate. This keeps the
        sequence nested: H(k) is always a subset of H(k + 1) whenever both
        counts are requested.
        """
        import pandas as pd
        from sklearn.cluster import KMeans

        os.makedirs(save_root, exist_ok=True)
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

        if hub_counts is None:
            hub_counts = getattr(config, "MEITUAN_SERIES_HUB_COUNTS", [100, 90, 80, 70, 60, 50, 40, 30, 20])
        hub_counts = sorted({int(h) for h in hub_counts}, reverse=True)
        if not hub_counts or min(hub_counts) <= 0:
            raise ValueError("hub_counts must contain positive integers.")

        parent_hubs = max(hub_counts)
        min_hubs = min(hub_counts)

        def _schedule_values(value, label):
            if isinstance(value, dict):
                missing = [h for h in hub_counts if h not in value]
                if missing:
                    raise ValueError(f"{label} schedule is missing hub counts: {missing}")
                return {h: int(value[h]) for h in hub_counts}
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return {h: int(value[0]) for h in hub_counts}
                if len(value) != len(hub_counts):
                    raise ValueError(
                        f"{label} schedule length {len(value)} does not match hub_counts length {len(hub_counts)}."
                    )
                return {h: int(v) for h, v in zip(hub_counts, value)}
            return {h: int(value) for h in hub_counts}

        base_schedule = _schedule_values(num_bases, "num_bases")
        swap_schedule = _schedule_values(num_swapping, "num_swapping")
        max_bases = max(base_schedule.values())
        max_swapping = max(swap_schedule.values())

        if min(base_schedule.values()) <= 0 or min(swap_schedule.values()) <= 0:
            raise ValueError("num_bases and num_swapping schedules must be positive.")

        if colocate_base_swaps is None:
            colocate_base_swaps = bool(getattr(config, "MEITUAN_COLOCATE_BASE_SWAPS", True))

        if colocate_base_swaps:
            for h_count in hub_counts:
                if swap_schedule[h_count] < base_schedule[h_count]:
                    raise ValueError(
                        f"num_swapping ({swap_schedule[h_count]}) must be >= num_bases "
                        f"({base_schedule[h_count]}) for H={h_count} when colocate_base_swaps=True."
                    )

        if map_half_size_km is None:
            map_half_size_km = config.MEITUAN_MAP_HALF_SIZE_KM

        def _date_list(value, fallback):
            if value is None:
                value = fallback
            if isinstance(value, (list, tuple, set)):
                values = [int(v) for v in value]
            else:
                values = [int(value)]
            if not values:
                if isinstance(fallback, (list, tuple, set)):
                    values = [int(v) for v in fallback]
                else:
                    values = [int(fallback)]
            return sorted(set(values))

        def _get_config_dates(name, fallback):
            value = getattr(config, name, None)
            return _date_list(value, fallback) if value else _date_list(fallback, fallback)

        candidate_dates = _date_list(
            candidate_date_filter,
            _get_config_dates("MEITUAN_CANDIDATE_DATES", date_filter),
        )
        demand_dates = _date_list(
            demand_date_filter,
            _get_config_dates("MEITUAN_DEMAND_DATES", date_filter),
        )

        if candidate_hour_start is None:
            candidate_hour_start = int(getattr(config, "MEITUAN_CANDIDATE_HOUR_START", config.MEITUAN_WORK_HOUR_START))
        if candidate_hour_end is None:
            candidate_hour_end = int(getattr(config, "MEITUAN_CANDIDATE_HOUR_END", config.MEITUAN_WORK_HOUR_END + 1))
        if demand_hour_start is None:
            demand_hour_start = int(getattr(config, "MEITUAN_DEMAND_HOUR", 12))
        if demand_hour_end is None:
            demand_hour_end = demand_hour_start + 1

        def _hour_duration(start_hour, end_hour):
            duration = (int(end_hour) - int(start_hour)) % 24
            if duration <= 0:
                raise ValueError("Hour windows must have positive duration under [start, end) semantics.")
            return duration

        candidate_hour_count = _hour_duration(candidate_hour_start, candidate_hour_end)
        demand_hour_count = _hour_duration(demand_hour_start, demand_hour_end)

        def _filter_period(frame, dates, start_hour, end_hour):
            subset = frame[frame["dt"].isin(dates)].copy()
            if subset.empty:
                return subset

            order_ts = pd.to_datetime(subset["platform_order_time"], unit="s", utc=True).dt.tz_convert(config.MEITUAN_TIMEZONE)
            subset["order_hour"] = order_ts.dt.hour

            if start_hour < end_hour:
                mask = (subset["order_hour"] >= start_hour) & (subset["order_hour"] < end_hour)
            else:
                mask = (subset["order_hour"] >= start_hour) | (subset["order_hour"] < end_hour)
            return subset[mask].copy()

        def _project(frame, lon_center, lat_center):
            projected = frame.copy()
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * np.cos(lat_center * np.pi / 180)
            projected["sender_x_km"] = (projected["sender_lng"] / 1e6 - lon_center) * km_per_deg_lon
            projected["sender_y_km"] = (projected["sender_lat"] / 1e6 - lat_center) * km_per_deg_lat
            projected["recipient_x_km"] = (projected["recipient_lng"] / 1e6 - lon_center) * km_per_deg_lon
            projected["recipient_y_km"] = (projected["recipient_lat"] / 1e6 - lat_center) * km_per_deg_lat
            return projected

        def _clip_area(frame):
            return frame[
                frame["sender_x_km"].between(-map_half_size_km, map_half_size_km)
                & frame["sender_y_km"].between(-map_half_size_km, map_half_size_km)
                & frame["recipient_x_km"].between(-map_half_size_km, map_half_size_km)
                & frame["recipient_y_km"].between(-map_half_size_km, map_half_size_km)
            ].copy()

        def _weighted_kmeans(points, weights, n_clusters, label):
            if n_clusters <= 0:
                return []
            if len(points) < n_clusters:
                raise ValueError(
                    f"Cannot generate {n_clusters} {label} candidates from only {len(points)} seed points."
                )
            if n_clusters == 1:
                weights_arr = np.asarray(weights, dtype=float)
                weights_arr = weights_arr / max(weights_arr.sum(), 1.0)
                center = np.average(np.asarray(points, dtype=float), axis=0, weights=weights_arr)
                return [tuple(float(v) for v in center)]

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(np.asarray(points, dtype=float), sample_weight=np.asarray(weights, dtype=float))
            ranked_labels = (
                pd.DataFrame({"label": labels, "weight": weights})
                .groupby("label")["weight"]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            return [
                (float(km.cluster_centers_[cluster][0]), float(km.cluster_centers_[cluster][1]))
                for cluster in ranked_labels
            ]

        def _assign_nearest(frame, x_col, y_col, hub_ids, center_arr):
            pts = frame[[x_col, y_col]].to_numpy(dtype=float)
            dist_sq = ((pts[:, None, :] - center_arr[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(dist_sq, axis=1)
            return [hub_ids[idx] for idx in nearest]

        print(
            f"Generating nested Meituan series: H={hub_counts}, "
            f"L={base_schedule}, E={swap_schedule}, demand_hour={demand_hour_start}"
        )
        df = pd.read_csv(raw_data_path)

        candidate_df = _filter_period(df, candidate_dates, candidate_hour_start, candidate_hour_end)
        if candidate_df.empty:
            raise ValueError(
                f"No candidate orders found for dates={candidate_dates}, hours=[{candidate_hour_start}, {candidate_hour_end})."
            )

        demand_df = _filter_period(df, demand_dates, demand_hour_start, demand_hour_end)
        if demand_df.empty:
            raise ValueError(
                f"No demand orders found for dates={demand_dates}, hours=[{demand_hour_start}, {demand_hour_end})."
            )

        top_poi = candidate_df["poi_id"].value_counts().idxmax()
        center_node = candidate_df[candidate_df["poi_id"] == top_poi].iloc[0]
        lon_center = center_node["sender_lng"] / 1e6
        lat_center = center_node["sender_lat"] / 1e6

        candidate_area = _clip_area(_project(candidate_df, lon_center, lat_center))
        demand_area = _clip_area(_project(demand_df, lon_center, lat_center))

        if candidate_area.empty:
            raise ValueError(
                f"No candidate orders in map window half-size {map_half_size_km}km."
            )
        if demand_area.empty:
            raise ValueError(
                f"No demand orders in map window half-size {map_half_size_km}km."
            )

        p_stats = candidate_area.groupby(
            ["sender_lng", "sender_lat", "sender_x_km", "sender_y_km"]
        ).size().reset_index(name="activity_count")
        d_stats = candidate_area.groupby(
            ["recipient_lng", "recipient_lat", "recipient_x_km", "recipient_y_km"]
        ).size().reset_index(name="activity_count")
        p_stats.columns = ["lng", "lat", "x_km", "y_km", "activity_count"]
        d_stats.columns = ["lng", "lat", "x_km", "y_km", "activity_count"]

        node_stats = pd.concat([p_stats, d_stats]).groupby(
            ["lng", "lat", "x_km", "y_km"]
        )["activity_count"].sum().reset_index()

        if len(node_stats) < parent_hubs:
            raise ValueError(
                f"Cannot generate parent H={parent_hubs} from only {len(node_stats)} unique candidate points."
            )

        h_xy = node_stats[["x_km", "y_km"]].to_numpy()
        h_w = node_stats["activity_count"].to_numpy()
        h_kmeans = KMeans(n_clusters=parent_hubs, random_state=42, n_init="auto")
        node_stats["cluster"] = h_kmeans.fit_predict(h_xy, sample_weight=h_w)

        h_cluster_rank = (
            node_stats.groupby("cluster")["activity_count"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        h_cluster_to_id = {c: f"H{i + 1:03d}" for i, c in enumerate(h_cluster_rank)}
        h_activity = node_stats.groupby("cluster")["activity_count"].sum().to_dict()

        hub_ids = []
        h_id_to_xy = {}
        h_id_to_activity = {}
        for cluster_id in h_cluster_rank:
            h_id = h_cluster_to_id[cluster_id]
            hub_ids.append(h_id)
            h_id_to_xy[h_id] = (
                float(h_kmeans.cluster_centers_[cluster_id][0]),
                float(h_kmeans.cluster_centers_[cluster_id][1]),
            )
            h_id_to_activity[h_id] = float(h_activity[cluster_id])

        center_arr = np.asarray([h_id_to_xy[h] for h in hub_ids], dtype=float)

        demand_parent = demand_area.copy()
        demand_parent["p_parent"] = _assign_nearest(demand_parent, "sender_x_km", "sender_y_km", hub_ids, center_arr)
        demand_parent["d_parent"] = _assign_nearest(demand_parent, "recipient_x_km", "recipient_y_km", hub_ids, center_arr)
        endpoint_counts = {h: 0.0 for h in hub_ids}
        for h_id, count in demand_parent["p_parent"].value_counts().items():
            endpoint_counts[h_id] += float(count)
        for h_id, count in demand_parent["d_parent"].value_counts().items():
            endpoint_counts[h_id] += float(count)

        parent_pair_orders = (
            demand_parent[demand_parent["p_parent"] != demand_parent["d_parent"]]
            .groupby(["p_parent", "d_parent"])
            .size()
            .reset_index(name="order_count")
        )
        demand_denominator = float(len(demand_dates) * demand_hour_count)
        parent_pair_orders["demand"] = parent_pair_orders["order_count"] / demand_denominator

        hub_points = [h_id_to_xy[h] for h in hub_ids]
        hub_weights = [h_id_to_activity[h] for h in hub_ids]

        base_pool = _weighted_kmeans(hub_points, hub_weights, max_bases, "base")

        candidate_mapped = candidate_area.copy()
        candidate_mapped["p_id"] = _assign_nearest(candidate_mapped, "sender_x_km", "sender_y_km", hub_ids, center_arr)
        candidate_mapped["d_id"] = _assign_nearest(candidate_mapped, "recipient_x_km", "recipient_y_km", hub_ids, center_arr)
        candidate_mapped = candidate_mapped[candidate_mapped["p_id"] != candidate_mapped["d_id"]]
        candidate_pairs = candidate_mapped.groupby(["p_id", "d_id"]).size().reset_index(name="count")

        swap_seed_points = list(hub_points)
        swap_seed_weights = list(hub_weights)
        for row in candidate_pairs.itertuples(index=False):
            p_xy = h_id_to_xy[row.p_id]
            d_xy = h_id_to_xy[row.d_id]
            swap_seed_points.append(((p_xy[0] + d_xy[0]) / 2.0, (p_xy[1] + d_xy[1]) / 2.0))
            swap_seed_weights.append(float(row.count))

        e_pool = []
        if colocate_base_swaps:
            e_pool.extend(base_pool[:max_bases])
        extra_swaps = max_swapping - len(e_pool)
        e_pool.extend(_weighted_kmeans(swap_seed_points, swap_seed_weights, extra_swaps, "swap"))

        if len(e_pool) < max_swapping:
            raise ValueError(f"Generated only {len(e_pool)} swap candidates; expected {max_swapping}.")

        activity_arr = np.asarray([h_id_to_activity[h] for h in hub_ids], dtype=float)
        endpoint_arr = np.asarray([endpoint_counts[h] for h in hub_ids], dtype=float)
        activity_norm = activity_arr / max(float(activity_arr.mean()), 1.0)
        endpoint_norm = endpoint_arr / max(float(endpoint_arr[endpoint_arr > 0].mean()) if np.any(endpoint_arr > 0) else 1.0, 1.0)
        hierarchy_load = {
            h_id: float(max(1e-6, 0.5 * activity_norm[i] + 0.5 * endpoint_norm[i]))
            for i, h_id in enumerate(hub_ids)
        }

        def _geo_scale():
            points = np.asarray(hub_points, dtype=float)
            if len(points) <= 1:
                return 1.0
            dists = []
            for i in range(len(points)):
                diff = points[i + 1:] - points[i]
                if len(diff):
                    dists.extend(np.sqrt((diff ** 2).sum(axis=1)).tolist())
            positive = [d for d in dists if d > 1e-12]
            return float(np.median(positive)) if positive else 1.0

        geo_scale = _geo_scale()

        def _make_initial_clusters():
            return {
                h_id: {
                    "cluster_id": h_id,
                    "members": [h_id],
                    "rep": h_id,
                    "x": h_id_to_xy[h_id][0],
                    "y": h_id_to_xy[h_id][1],
                    "load": hierarchy_load[h_id],
                    "activity": h_id_to_activity[h_id],
                    "endpoint_orders": endpoint_counts[h_id],
                }
                for h_id in hub_ids
            }

        def _snapshot(active_clusters):
            parent_to_rep = {}
            rep_to_cluster = {}
            for cluster in active_clusters.values():
                rep = cluster["rep"]
                rep_to_cluster[rep] = {
                    "members": list(cluster["members"]),
                    "load": float(cluster["load"]),
                    "activity": float(cluster["activity"]),
                    "endpoint_orders": float(cluster["endpoint_orders"]),
                    "x": float(cluster["x"]),
                    "y": float(cluster["y"]),
                }
                for member in cluster["members"]:
                    parent_to_rep[member] = rep
            retained = [h_id for h_id in hub_ids if h_id in rep_to_cluster]
            return {
                "retained": retained,
                "parent_to_rep": parent_to_rep,
                "rep_to_cluster": rep_to_cluster,
            }

        active = _make_initial_clusters()
        snapshots = {parent_hubs: _snapshot(active)}
        total_load = sum(cluster["load"] for cluster in active.values())
        merge_history = []
        merge_step = 0

        while len(active) > min_hubs:
            next_count = len(active) - 1
            target_load = total_load / float(next_count)
            keys = list(active.keys())
            best_score = None
            best_pair = None
            for i, key_a in enumerate(keys[:-1]):
                a = active[key_a]
                for key_b in keys[i + 1:]:
                    b = active[key_b]
                    dx = a["x"] - b["x"]
                    dy = a["y"] - b["y"]
                    geo = float(np.sqrt(dx * dx + dy * dy) / max(geo_scale, 1e-9))
                    load_diff = abs(np.log1p(a["load"]) - np.log1p(b["load"]))
                    combined_ratio = (a["load"] + b["load"]) / max(target_load, 1e-9)
                    overload = max(0.0, combined_ratio - float(hierarchy_overload_tolerance))
                    score = (
                        geo
                        + float(hierarchy_activity_weight) * load_diff
                        + float(hierarchy_balance_weight) * overload * overload
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (key_a, key_b)

            key_a, key_b = best_pair
            a = active.pop(key_a)
            b = active.pop(key_b)
            merged_load = a["load"] + b["load"]
            merged_activity = a["activity"] + b["activity"]
            merged_endpoint_orders = a["endpoint_orders"] + b["endpoint_orders"]
            x_val = (a["x"] * a["load"] + b["x"] * b["load"]) / max(merged_load, 1e-9)
            y_val = (a["y"] * a["load"] + b["y"] * b["load"]) / max(merged_load, 1e-9)

            rep_candidates = [a["rep"], b["rep"]]
            rep = min(
                rep_candidates,
                key=lambda h: (
                    (h_id_to_xy[h][0] - x_val) ** 2 + (h_id_to_xy[h][1] - y_val) ** 2,
                    -h_id_to_activity[h],
                    h,
                ),
            )
            retire = b["rep"] if rep == a["rep"] else a["rep"]
            merge_step += 1
            new_key = f"C{merge_step:03d}"
            members = sorted(a["members"] + b["members"], key=lambda h: hub_ids.index(h))
            active[new_key] = {
                "cluster_id": new_key,
                "members": members,
                "rep": rep,
                "x": float(x_val),
                "y": float(y_val),
                "load": float(merged_load),
                "activity": float(merged_activity),
                "endpoint_orders": float(merged_endpoint_orders),
            }
            merge_history.append(
                {
                    "step": merge_step,
                    "from_count": next_count + 1,
                    "to_count": next_count,
                    "merged_a_rep": a["rep"],
                    "merged_b_rep": b["rep"],
                    "survivor_rep": rep,
                    "retired_rep": retire,
                    "merged_members": len(members),
                    "merged_activity": float(merged_activity),
                    "merged_endpoint_orders": float(merged_endpoint_orders),
                    "score": float(best_score),
                }
            )
            if next_count in hub_counts:
                snapshots[next_count] = _snapshot(active)

        period_id = f"h{demand_hour_start:02d}" if demand_hour_count == 1 else f"h{demand_hour_start:02d}_{demand_hour_count}h"
        period_orders_file = f"{period_id}.csv"

        def _aggregate_orders(snapshot):
            rows = {}
            for row in parent_pair_orders.itertuples(index=False):
                p = snapshot["parent_to_rep"][row.p_parent]
                d = snapshot["parent_to_rep"][row.d_parent]
                if p == d:
                    continue
                rows[(p, d)] = rows.get((p, d), 0.0) + float(row.demand)
            orders_rows = [
                {"start_node": p, "end_node": d, "demand": demand}
                for (p, d), demand in rows.items()
            ]
            return pd.DataFrame(orders_rows, columns=["start_node", "end_node", "demand"]).sort_values(
                ["start_node", "end_node"]
            )

        def _cluster_rows(snapshot, h_count):
            rows = []
            for rep in snapshot["retained"]:
                cluster = snapshot["rep_to_cluster"][rep]
                for member in cluster["members"]:
                    rows.append(
                        {
                            "h_count": h_count,
                            "parent_hub": member,
                            "retained_hub": rep,
                            "cluster_size": len(cluster["members"]),
                            "cluster_load": cluster["load"],
                            "cluster_activity": cluster["activity"],
                            "cluster_endpoint_orders": cluster["endpoint_orders"],
                        }
                    )
            return rows

        def _load_summary(values):
            arr = np.asarray(values, dtype=float)
            if len(arr) == 0:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "cv": 0.0}
            mean = float(arr.mean())
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": mean,
                "cv": float(arr.std(ddof=0) / mean) if mean > 0 else 0.0,
            }

        summary_rows = []
        for h_count in hub_counts:
            snapshot = snapshots[h_count]
            l_count = base_schedule[h_count]
            e_count = swap_schedule[h_count]
            name = f"Meituan_H{h_count:03d}_L{l_count:02d}_E{e_count:02d}_{period_id}"
            instance_dir = os.path.join(save_root, name)
            orders_dir = os.path.join(instance_dir, "orders")
            os.makedirs(orders_dir, exist_ok=True)

            locations = []
            for h_id in snapshot["retained"]:
                x_val, y_val = h_id_to_xy[h_id]
                locations.append({"index": h_id, "type": "H", "x": x_val, "y": y_val})
            for i, (x_val, y_val) in enumerate(base_pool[:l_count], 1):
                locations.append({"index": f"L{i}", "type": "L", "x": float(x_val), "y": float(y_val)})
            for i, (x_val, y_val) in enumerate(e_pool[:e_count], 1):
                locations.append({"index": f"E{i}", "type": "E", "x": float(x_val), "y": float(y_val)})

            orders_out = _aggregate_orders(snapshot)
            pd.DataFrame(locations).to_csv(os.path.join(instance_dir, "locations.csv"), index=False)
            orders_out.to_csv(os.path.join(instance_dir, "orders.csv"), index=False)
            orders_out.to_csv(os.path.join(orders_dir, period_orders_file), index=False)

            retained_clusters = [snapshot["rep_to_cluster"][rep] for rep in snapshot["retained"]]
            load_stats = _load_summary([cluster["load"] for cluster in retained_clusters])
            activity_stats = _load_summary([cluster["activity"] for cluster in retained_clusters])
            endpoint_stats = _load_summary([cluster["endpoint_orders"] for cluster in retained_clusters])
            cluster_sizes = [len(cluster["members"]) for cluster in retained_clusters]

            periods_row = {
                "period_id": period_id,
                "orders_file": os.path.join("orders", period_orders_file),
                "start_hour": demand_hour_start,
                "end_hour": demand_hour_end,
                "hour_semantics": "[start, end)",
                "dates": ";".join(str(d) for d in demand_dates),
                "num_dates": len(demand_dates),
                "duration_hours": demand_hour_count,
                "raw_orders": len(demand_df),
                "in_area_orders": len(demand_area),
                "parent_od_pairs": len(parent_pair_orders),
                "retained_od_pairs": len(orders_out),
                "total_hourly_demand": float(orders_out["demand"].sum()) if not orders_out.empty else 0.0,
            }
            pd.DataFrame([periods_row]).to_csv(os.path.join(instance_dir, "periods.csv"), index=False)
            pd.DataFrame(_cluster_rows(snapshot, h_count)).to_csv(os.path.join(instance_dir, "hierarchy_map.csv"), index=False)

            metadata = {
                "instance_name": name,
                "parent_hubs": parent_hubs,
                "hub_count": h_count,
                "base_count": l_count,
                "swapping_count": e_count,
                "retained_hubs": snapshot["retained"],
                "candidate_dates": candidate_dates,
                "candidate_hours": [candidate_hour_start, candidate_hour_end],
                "demand_dates": demand_dates,
                "demand_hours": [demand_hour_start, demand_hour_end],
                "map_half_size_km": map_half_size_km,
                "colocate_base_swaps": colocate_base_swaps,
                "hierarchy_activity_weight": hierarchy_activity_weight,
                "hierarchy_balance_weight": hierarchy_balance_weight,
                "hierarchy_overload_tolerance": hierarchy_overload_tolerance,
                "load_stats": load_stats,
                "activity_stats": activity_stats,
                "endpoint_order_stats": endpoint_stats,
                "cluster_size_stats": _load_summary(cluster_sizes),
            }
            with open(os.path.join(instance_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

            summary_row = {
                "instance_name": name,
                "instance_dir": instance_dir,
                "H": h_count,
                "L": l_count,
                "E": e_count,
                "period_id": period_id,
                "od_pairs": len(orders_out),
                "total_hourly_demand": float(orders_out["demand"].sum()) if not orders_out.empty else 0.0,
                "candidate_raw_orders": len(candidate_df),
                "candidate_in_area_orders": len(candidate_area),
                "raw_orders": len(demand_df),
                "in_area_orders": len(demand_area),
                "unique_candidate_points": len(node_stats),
                "parent_od_pairs": len(parent_pair_orders),
                "cluster_size_min": min(cluster_sizes) if cluster_sizes else 0,
                "cluster_size_max": max(cluster_sizes) if cluster_sizes else 0,
                "cluster_size_cv": _load_summary(cluster_sizes)["cv"],
                "cluster_load_min": load_stats["min"],
                "cluster_load_max": load_stats["max"],
                "cluster_load_cv": load_stats["cv"],
                "cluster_activity_min": activity_stats["min"],
                "cluster_activity_max": activity_stats["max"],
                "cluster_activity_cv": activity_stats["cv"],
                "cluster_endpoint_orders_min": endpoint_stats["min"],
                "cluster_endpoint_orders_max": endpoint_stats["max"],
                "cluster_endpoint_orders_cv": endpoint_stats["cv"],
            }
            summary_rows.append(summary_row)

            print(
                f"Generated {name}: H={h_count}, L={l_count}, E={e_count}, "
                f"OD={len(orders_out)}, demand={summary_row['total_hourly_demand']:.4f}/h, "
                f"cluster_load_cv={load_stats['cv']:.3f}"
            )

        pd.DataFrame(summary_rows).to_csv(os.path.join(save_root, "series_summary.csv"), index=False)
        pd.DataFrame(merge_history).to_csv(os.path.join(save_root, "merge_history.csv"), index=False)
        pd.DataFrame(
            [
                {
                    "parent_hub": h_id,
                    "x": h_id_to_xy[h_id][0],
                    "y": h_id_to_xy[h_id][1],
                    "activity_count": h_id_to_activity[h_id],
                    "endpoint_orders": endpoint_counts[h_id],
                    "hierarchy_load": hierarchy_load[h_id],
                }
                for h_id in hub_ids
            ]
        ).to_csv(os.path.join(save_root, "parent_hubs.csv"), index=False)

        root_metadata = {
            "raw_data_path": raw_data_path,
            "save_root": save_root,
            "hub_counts": hub_counts,
            "base_schedule": base_schedule,
            "swap_schedule": swap_schedule,
            "parent_hubs": parent_hubs,
            "candidate_dates": candidate_dates,
            "candidate_hours": [candidate_hour_start, candidate_hour_end],
            "candidate_duration_hours": candidate_hour_count,
            "candidate_raw_orders": len(candidate_df),
            "candidate_in_area_orders": len(candidate_area),
            "demand_dates": demand_dates,
            "demand_hours": [demand_hour_start, demand_hour_end],
            "demand_duration_hours": demand_hour_count,
            "demand_raw_orders": len(demand_df),
            "demand_in_area_orders": len(demand_area),
            "unique_candidate_points": len(node_stats),
            "parent_od_pairs": len(parent_pair_orders),
            "map_half_size_km": map_half_size_km,
            "colocate_base_swaps": colocate_base_swaps,
            "hierarchy_activity_weight": hierarchy_activity_weight,
            "hierarchy_balance_weight": hierarchy_balance_weight,
            "hierarchy_overload_tolerance": hierarchy_overload_tolerance,
        }
        with open(os.path.join(save_root, "series_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(root_metadata, f, indent=2, sort_keys=True)

        return summary_rows
