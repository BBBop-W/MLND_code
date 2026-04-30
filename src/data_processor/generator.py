import numpy as np
import random
import csv
import os
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
