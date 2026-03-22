import numpy as np
import random
import csv
import os
import math
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
        for i in range(1, config.NUM_PICKUP + 1):
            locations.append({"index": f"P{i}", "type": "P", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
        for i in range(1, config.NUM_DELIVERY + 1):
            locations.append({"index": f"D{i}", "type": "D", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
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
        pickups = [loc["index"] for loc in locations if loc["type"] == "P"]
        deliveries = [loc["index"] for loc in locations if loc["type"] == "D"]
        for p in pickups:
            for d in deliveries:
                if np.random.random() < config.DEMAND_PROBABILITY:
                    demand = np.random.uniform(config.DEMAND_MIN, config.DEMAND_MAX)
                    orders.append({"start_node": p, "end_node": d, "demand": demand})
                    
        with open(orders_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["start_node", "end_node", "demand"])
            writer.writeheader()
            writer.writerows(orders)
        print(f"Synthetic instance generated and saved to: {save_dir}")

    def generate_meituan_hourly(self, raw_data_path: str, save_dir: str, 
                                num_pickup: int = 100, num_delivery: int = 100, 
                                num_bases: int = 25, num_swapping: int = 50,
                                date_filter: int = 20221017,
                                map_half_size_km: float = None,
                                min_node_hourly_orders: float = None):
        import pandas as pd
        from sklearn.cluster import KMeans
        os.makedirs(save_dir, exist_ok=True)

        if num_swapping < num_bases:
            raise ValueError(
                f"num_swapping ({num_swapping}) must be >= num_bases ({num_bases}) to ensure each base has a colocated swap station."
            )

        if map_half_size_km is None:
            map_half_size_km = config.MEITUAN_MAP_HALF_SIZE_KM
        if min_node_hourly_orders is None:
            min_node_hourly_orders = config.MEITUAN_MIN_NODE_HOURLY_ORDERS

        work_hour_start = config.MEITUAN_WORK_HOUR_START
        work_hour_end = config.MEITUAN_WORK_HOUR_END
        work_hours_count = work_hour_end - work_hour_start + 1
        if work_hours_count <= 0:
            raise ValueError("Invalid work-hour range: end must be >= start.")
        
        print(f"Reading data from {raw_data_path} for date {date_filter}...")
        df = pd.read_csv(raw_data_path)
        df = df[df['dt'] == date_filter].copy()
        
        if df.empty:
            raise ValueError(f"No data found for date {date_filter}")

        # Keep only work-hour orders for realistic hourly averaging.
        order_ts = pd.to_datetime(df['platform_order_time'], unit='s', utc=True).dt.tz_convert(config.MEITUAN_TIMEZONE)
        df['order_hour'] = order_ts.dt.hour
        work_df = df[df['order_hour'].between(work_hour_start, work_hour_end)].copy()
        if work_df.empty:
            raise ValueError(
                f"No orders found in work hours [{work_hour_start}, {work_hour_end}] for date {date_filter}."
            )

        # Define center and project coordinates to km (strict unit consistency)
        top_poi = work_df['poi_id'].value_counts().idxmax()
        center_node = work_df[work_df['poi_id'] == top_poi].iloc[0]
        lon_center = center_node['sender_lng'] / 1e6
        lat_center = center_node['sender_lat'] / 1e6

        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(lat_center * np.pi / 180)

        work_df['sender_x_km'] = (work_df['sender_lng'] / 1e6 - lon_center) * km_per_deg_lon
        work_df['sender_y_km'] = (work_df['sender_lat'] / 1e6 - lat_center) * km_per_deg_lat
        work_df['recipient_x_km'] = (work_df['recipient_lng'] / 1e6 - lon_center) * km_per_deg_lon
        work_df['recipient_y_km'] = (work_df['recipient_lat'] / 1e6 - lat_center) * km_per_deg_lat

        # Scale by map size: keep orders whose pickup and delivery both fall in the selected map window.
        area_df = work_df[
            work_df['sender_x_km'].between(-map_half_size_km, map_half_size_km)
            & work_df['sender_y_km'].between(-map_half_size_km, map_half_size_km)
            & work_df['recipient_x_km'].between(-map_half_size_km, map_half_size_km)
            & work_df['recipient_y_km'].between(-map_half_size_km, map_half_size_km)
        ].copy()

        if area_df.empty:
            raise ValueError(
                f"No orders in map window half-size {map_half_size_km}km for date {date_filter}."
            )

        # Real-node activity in selected map (work hours only).
        p_stats = area_df.groupby(['sender_lng', 'sender_lat', 'sender_x_km', 'sender_y_km']).size().reset_index(name='daily_count')
        d_stats = area_df.groupby(['recipient_lng', 'recipient_lat', 'recipient_x_km', 'recipient_y_km']).size().reset_index(name='daily_count')
        p_stats['hourly_avg'] = p_stats['daily_count'] / float(work_hours_count)
        d_stats['hourly_avg'] = d_stats['daily_count'] / float(work_hours_count)

        p_filtered = p_stats[p_stats['hourly_avg'] >= min_node_hourly_orders].copy()
        d_filtered = d_stats[d_stats['hourly_avg'] >= min_node_hourly_orders].copy()

        # Fallback if threshold is too strict.
        if p_filtered.empty:
            p_filtered = p_stats.copy()
        if d_filtered.empty:
            d_filtered = d_stats.copy()

        # Aggregate raw nodes to requested PD counts (cluster in km space, weighted by work-hour order frequency).
        n_p = max(1, min(num_pickup, len(p_filtered)))
        n_d = max(1, min(num_delivery, len(d_filtered)))

        p_xy = p_filtered[['sender_x_km', 'sender_y_km']].to_numpy()
        p_w = p_filtered['daily_count'].to_numpy()
        p_kmeans = KMeans(n_clusters=n_p, random_state=42, n_init="auto")
        p_filtered['cluster'] = p_kmeans.fit_predict(p_xy, sample_weight=p_w)

        d_xy = d_filtered[['recipient_x_km', 'recipient_y_km']].to_numpy()
        d_w = d_filtered['daily_count'].to_numpy()
        d_kmeans = KMeans(n_clusters=n_d, random_state=42, n_init="auto")
        d_filtered['cluster'] = d_kmeans.fit_predict(d_xy, sample_weight=d_w)

        p_cluster_rank = (
            p_filtered.groupby('cluster')['daily_count']
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        d_cluster_rank = (
            d_filtered.groupby('cluster')['daily_count']
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        p_cluster_to_id = {c: f"P{i + 1}" for i, c in enumerate(p_cluster_rank)}
        d_cluster_to_id = {c: f"D{i + 1}" for i, c in enumerate(d_cluster_rank)}

        p_map = p_filtered[['sender_lng', 'sender_lat', 'cluster']].copy()
        d_map = d_filtered[['recipient_lng', 'recipient_lat', 'cluster']].copy()

        # Keep only orders whose endpoints survived frequency filtering, then aggregate by clustered IDs.
        selected_orders = area_df.merge(p_map, on=['sender_lng', 'sender_lat'], how='inner')
        selected_orders = selected_orders.merge(d_map, on=['recipient_lng', 'recipient_lat'], how='inner', suffixes=('_p', '_d'))

        if selected_orders.empty:
            raise ValueError(
                "Selected pickup/delivery nodes produce no orders. Increase map size or lower min_node_hourly_orders."
            )

        selected_orders['p_id'] = selected_orders['cluster_p'].map(p_cluster_to_id)
        selected_orders['d_id'] = selected_orders['cluster_d'].map(d_cluster_to_id)

        pair_orders = selected_orders.groupby(['p_id', 'd_id']).size().reset_index(name='daily_count')
        # OD demand is the hourly average over configured work hours.
        pair_orders['demand'] = pair_orders['daily_count'] / float(work_hours_count)

        # Prune unused clusters and compact IDs after OD aggregation.
        active_p = sorted(pair_orders['p_id'].unique(), key=lambda x: int(x[1:]))
        active_d = sorted(pair_orders['d_id'].unique(), key=lambda x: int(x[1:]))
        p_old_to_new = {old: f"P{i + 1}" for i, old in enumerate(active_p)}
        d_old_to_new = {old: f"D{i + 1}" for i, old in enumerate(active_d)}

        pair_orders['start_node'] = pair_orders['p_id'].map(p_old_to_new)
        pair_orders['end_node'] = pair_orders['d_id'].map(d_old_to_new)

        # Save simulation events mapped to compact node IDs.
        sim_events = selected_orders.copy()
        sim_events['p_id'] = sim_events['p_id'].map(p_old_to_new)
        sim_events['d_id'] = sim_events['d_id'].map(d_old_to_new)
        sim_events = sim_events.dropna(subset=['p_id', 'd_id'])
        sim_events[['platform_order_time', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat', 'p_id', 'd_id']].to_csv(
            os.path.join(save_dir, "simulation_events.csv"), index=False
        )

        # Build aggregated P/D locations from cluster centers (km).
        locations = []
        p_id_to_xy = {}
        for cluster_id, p_id_old in p_cluster_to_id.items():
            if p_id_old not in p_old_to_new:
                continue
            p_id = p_old_to_new[p_id_old]
            x_val = float(p_kmeans.cluster_centers_[cluster_id][0])
            y_val = float(p_kmeans.cluster_centers_[cluster_id][1])
            locations.append({"index": p_id, "type": "P", "x": x_val, "y": y_val})
            p_id_to_xy[p_id] = (x_val, y_val)

        d_id_to_xy = {}
        for cluster_id, d_id_old in d_cluster_to_id.items():
            if d_id_old not in d_old_to_new:
                continue
            d_id = d_old_to_new[d_id_old]
            x_val = float(d_kmeans.cluster_centers_[cluster_id][0])
            y_val = float(d_kmeans.cluster_centers_[cluster_id][1])
            locations.append({"index": d_id, "type": "D", "x": x_val, "y": y_val})
            d_id_to_xy[d_id] = (x_val, y_val)

        # Place L/E farther from PD based on configured strategy.
        base_locations = []
        placement_mode = getattr(config, "MEITUAN_LE_PLACEMENT_MODE", "centroid").lower()
        jitter_radius_km = max(0.75, float(getattr(config, "MEITUAN_LE_JITTER_RADIUS_KM", 0.75)))
        spread_km = float(getattr(config, "MEITUAN_LE_CENTROID_SPREAD_KM", 1.5))
        min_ee_distance_km = float(getattr(config, "MEITUAN_MIN_EE_DISTANCE_KM", 1.0))

        def _clamp_xy(x_val, y_val):
            x_val = float(np.clip(x_val, -map_half_size_km, map_half_size_km))
            y_val = float(np.clip(y_val, -map_half_size_km, map_half_size_km))
            return x_val, y_val

        def _min_dist_to_existing(pt, existing_pts):
            if not existing_pts:
                return float('inf')
            px, py = pt
            return min(np.hypot(px - ex, py - ey) for ex, ey in existing_pts)

        def _separate_e_candidate(x_seed, y_seed, existing_e_pts, attempts=200):
            # Try local jitters around seed first, then random fallback in map bounds.
            best = _clamp_xy(x_seed, y_seed)
            best_score = _min_dist_to_existing(best, existing_e_pts)
            if best_score >= min_ee_distance_km:
                return best

            local_radius = max(min_ee_distance_km, 0.5)
            for _ in range(attempts):
                theta = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0.2 * local_radius, local_radius)
                cand = _clamp_xy(x_seed + radius * np.cos(theta), y_seed + radius * np.sin(theta))
                score = _min_dist_to_existing(cand, existing_e_pts)
                if score > best_score:
                    best, best_score = cand, score
                if score >= min_ee_distance_km:
                    return cand

            for _ in range(attempts):
                cand = (
                    float(np.random.uniform(-map_half_size_km, map_half_size_km)),
                    float(np.random.uniform(-map_half_size_km, map_half_size_km)),
                )
                score = _min_dist_to_existing(cand, existing_e_pts)
                if score > best_score:
                    best, best_score = cand, score
                if score >= min_ee_distance_km:
                    return cand

            return best

        if placement_mode == "centroid":
            # Weighted geometric centroid of original work-hour orders (origins) for L placement.
            p_weights = p_filtered['daily_count'].to_numpy(dtype=float)
            p_total_w = max(float(p_weights.sum()), 1.0)
            l_cx = float((p_filtered['sender_x_km'] * p_weights).sum() / p_total_w)
            l_cy = float((p_filtered['sender_y_km'] * p_weights).sum() / p_total_w)

            # Ensure L-ring spacing implies E1..E_num_bases are not too close.
            if num_bases > 1:
                min_radius_for_spacing = min_ee_distance_km / (2.0 * np.sin(np.pi / num_bases))
            else:
                min_radius_for_spacing = 0.0
            ring_radius_l = min(max(spread_km, min_radius_for_spacing, 0.5), map_half_size_km * 0.75)
            for i in range(num_bases):
                theta = (2.0 * np.pi * i) / max(num_bases, 1)
                x_val = l_cx + ring_radius_l * np.cos(theta)
                y_val = l_cy + ring_radius_l * np.sin(theta)
                x_val, y_val = _clamp_xy(x_val, y_val)
                base_locations.append((x_val, y_val))
                locations.append({"index": f"L{i + 1}", "type": "L", "x": x_val, "y": y_val})
        else:
            pickup_hotspots = list(p_id_to_xy.values())
            for i in range(num_bases):
                if i < len(pickup_hotspots):
                    x_val, y_val = pickup_hotspots[i]
                    theta = np.random.uniform(0, 2 * np.pi)
                    x_val += jitter_radius_km * np.cos(theta)
                    y_val += jitter_radius_km * np.sin(theta)
                else:
                    x_val = np.random.uniform(-map_half_size_km, map_half_size_km)
                    y_val = np.random.uniform(-map_half_size_km, map_half_size_km)
                x_val, y_val = _clamp_xy(x_val, y_val)
                base_locations.append((x_val, y_val))
                locations.append({"index": f"L{i + 1}", "type": "L", "x": x_val, "y": y_val})

        # Ensure each base has one colocated swapping station.
        existing_e_points = []
        for i, (x_val, y_val) in enumerate(base_locations, 1):
            locations.append({"index": f"E{i}", "type": "E", "x": x_val, "y": y_val})
            existing_e_points.append((x_val, y_val))

        # Extra swapping stations.
        delivery_hotspots = list(d_id_to_xy.values())
        d_weights = d_filtered['daily_count'].to_numpy(dtype=float)
        d_total_w = max(float(d_weights.sum()), 1.0)
        e_cx = float((d_filtered['recipient_x_km'] * d_weights).sum() / d_total_w)
        e_cy = float((d_filtered['recipient_y_km'] * d_weights).sum() / d_total_w)
        ring_radius_e = min(max(spread_km, 0.5), map_half_size_km * 0.75)

        for i in range(num_bases + 1, num_swapping + 1):
            idx = i - (num_bases + 1)
            if placement_mode == "centroid":
                theta = (2.0 * np.pi * idx) / max(num_swapping - num_bases, 1)
                x_val = e_cx + ring_radius_e * np.cos(theta)
                y_val = e_cy + ring_radius_e * np.sin(theta)
            elif idx < len(delivery_hotspots):
                x_val, y_val = delivery_hotspots[idx]
                theta = np.random.uniform(0, 2 * np.pi)
                x_val += jitter_radius_km * np.cos(theta)
                y_val += jitter_radius_km * np.sin(theta)
            else:
                x_val = np.random.uniform(-map_half_size_km, map_half_size_km)
                y_val = np.random.uniform(-map_half_size_km, map_half_size_km)
            x_val, y_val = _separate_e_candidate(x_val, y_val, existing_e_points)
            locations.append({"index": f"E{i}", "type": "E", "x": x_val, "y": y_val})
            existing_e_points.append((x_val, y_val))

        pd.DataFrame(locations).to_csv(os.path.join(save_dir, "locations.csv"), index=False)

        orders_out = pair_orders[['start_node', 'end_node', 'demand']].copy()
        orders_out.to_csv(os.path.join(save_dir, "orders.csv"), index=False)

        loc_df = pd.DataFrame(locations)
        p_coords = set(zip(loc_df[loc_df['type'] == 'P']['x'].round(6), loc_df[loc_df['type'] == 'P']['y'].round(6)))
        e_coords = set(zip(loc_df[loc_df['type'] == 'E']['x'].round(6), loc_df[loc_df['type'] == 'E']['y'].round(6)))
        overlap_pe = p_coords & e_coords

        total_unique_p_day = df.groupby(['sender_lng', 'sender_lat']).size().shape[0]
        total_unique_d_day = df.groupby(['recipient_lng', 'recipient_lat']).size().shape[0]
        total_pairs_day = df.groupby(['sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']).size().shape[0]

        total_unique_p_work = work_df.groupby(['sender_lng', 'sender_lat']).size().shape[0]
        total_unique_d_work = work_df.groupby(['recipient_lng', 'recipient_lat']).size().shape[0]
        total_pairs_work = work_df.groupby(['sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']).size().shape[0]

        area_unique_p = area_df.groupby(['sender_lng', 'sender_lat']).size().shape[0]
        area_unique_d = area_df.groupby(['recipient_lng', 'recipient_lat']).size().shape[0]
        area_pairs = area_df.groupby(['sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']).size().shape[0]

        print(
            f"Meituan real-hourly instance generated at: {save_dir}\n"
            f"  aggregation mode: clustered to targets (P={num_pickup}, D={num_delivery})\n"
            f"  L/E placement mode: {placement_mode}\n"
            f"  work-hour window: {work_hour_start}:00-{work_hour_end}:59 ({work_hours_count}h), tz={config.MEITUAN_TIMEZONE}\n"
            f"  map half-size: {map_half_size_km:.2f} km (side {2 * map_half_size_km:.2f} km)\n"
            f"  raw(date) orders={len(df)}, unique origins={total_unique_p_day}, unique destinations={total_unique_d_day}, unique OD pairs={total_pairs_day}\n"
            f"  work-hour orders={len(work_df)}, unique origins={total_unique_p_work}, unique destinations={total_unique_d_work}, unique OD pairs={total_pairs_work}\n"
            f"  in-map orders={len(area_df)}, unique origins={area_unique_p}, unique destinations={area_unique_d}, unique OD pairs={area_pairs}\n"
            f"  after node frequency filter >= {min_node_hourly_orders:.3f}/h: P candidates={len(p_filtered)}, D candidates={len(d_filtered)}\n"
            f"  clustered nodes before OD pruning: P={n_p}, D={n_d}; after OD pruning: P={len(p_id_to_xy)}, D={len(d_id_to_xy)}\n"
            f"  selected nodes: P={len(p_id_to_xy)}, D={len(d_id_to_xy)}, L={num_bases}, E={num_swapping}\n"
            f"  retained OD pairs: {len(orders_out)}, total clustered demand(per-hour)={orders_out['demand'].sum():.4f}\n"
            f"  min E-E target distance={min_ee_distance_km:.2f} km\n"
            f"  P-E coordinate overlap count={len(overlap_pe)}"
        )

        if overlap_pe:
            examples = list(overlap_pe)[:5]
            print(f"  overlap examples (x_km,y_km): {examples}")
            print(f"  note: E1..E{num_bases} are intentionally colocated with L. P-E overlap above is unintended coincidence/data geometry.")
