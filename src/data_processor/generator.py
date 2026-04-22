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
        
        # Rename columns to uniform names for merging
        p_stats.columns = ['lng', 'lat', 'x_km', 'y_km', 'daily_count']
        d_stats.columns = ['lng', 'lat', 'x_km', 'y_km', 'daily_count']
        
        # Combine all node activities
        node_stats = pd.concat([p_stats, d_stats]).groupby(['lng', 'lat', 'x_km', 'y_km'])['daily_count'].sum().reset_index()
        node_stats['hourly_avg'] = node_stats['daily_count'] / float(work_hours_count)

        h_filtered = node_stats[node_stats['hourly_avg'] >= min_node_hourly_orders].copy()

        # Fallback if threshold is too strict.
        if h_filtered.empty:
            h_filtered = node_stats.copy()

        # Aggregate raw nodes to requested Hub counts (cluster in km space, weighted by work-hour order frequency).
        n_h = max(1, min(num_hubs, len(h_filtered)))

        h_xy = h_filtered[['x_km', 'y_km']].to_numpy()
        h_w = h_filtered['daily_count'].to_numpy()
        h_kmeans = KMeans(n_clusters=n_h, random_state=42, n_init="auto")
        h_filtered['cluster'] = h_kmeans.fit_predict(h_xy, sample_weight=h_w)

        h_cluster_rank = (
            h_filtered.groupby('cluster')['daily_count']
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        
        h_cluster_to_id = {c: f"H{i + 1}" for i, c in enumerate(h_cluster_rank)}

        h_map = h_filtered[['lng', 'lat', 'cluster']].copy()

        # Merge for senders
        selected_orders = area_df.merge(h_map, left_on=['sender_lng', 'sender_lat'], right_on=['lng', 'lat'], how='inner')
        selected_orders = selected_orders.rename(columns={'cluster': 'cluster_p'})
        # Merge for recipients
        selected_orders = selected_orders.merge(h_map, left_on=['recipient_lng', 'recipient_lat'], right_on=['lng', 'lat'], how='inner')
        selected_orders = selected_orders.rename(columns={'cluster': 'cluster_d'})

        if selected_orders.empty:
            raise ValueError(
                "Selected hub nodes produce no orders. Increase map size or lower min_node_hourly_orders."
            )

        selected_orders['p_id'] = selected_orders['cluster_p'].map(h_cluster_to_id)
        selected_orders['d_id'] = selected_orders['cluster_d'].map(h_cluster_to_id)

        pair_orders = selected_orders.groupby(['p_id', 'd_id']).size().reset_index(name='daily_count')
        # OD demand is the hourly average over configured work hours.
        pair_orders['demand'] = pair_orders['daily_count'] / float(work_hours_count)

        # Prune unused clusters and compact IDs after OD aggregation.
        # Find uniquely used IDs (union of p_id and d_id)
        active_hubs = sorted(set(pair_orders['p_id'].unique()) | set(pair_orders['d_id'].unique()), key=lambda x: int(x[1:]))
        h_old_to_new = {old: f"H{i + 1}" for i, old in enumerate(active_hubs)}

        pair_orders['start_node'] = pair_orders['p_id'].map(h_old_to_new)
        pair_orders['end_node'] = pair_orders['d_id'].map(h_old_to_new)

        # Remove self-loops where p == d if any
        pair_orders = pair_orders[pair_orders['start_node'] != pair_orders['end_node']]

        # Save simulation events mapped to compact node IDs.
        sim_events = selected_orders.copy()
        sim_events['p_id'] = sim_events['p_id'].map(h_old_to_new)
        sim_events['d_id'] = sim_events['d_id'].map(h_old_to_new)
        sim_events = sim_events.dropna(subset=['p_id', 'd_id'])
        sim_events = sim_events[sim_events['p_id'] != sim_events['d_id']]
        sim_events[['platform_order_time', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat', 'p_id', 'd_id']].to_csv(
            os.path.join(save_dir, "simulation_events.csv"), index=False
        )

        # Build aggregated Hub locations from cluster centers (km).
        locations = []
        h_id_to_xy = {}
        for cluster_id, h_id_old in h_cluster_to_id.items():
            if h_id_old not in h_old_to_new:
                continue
            h_id = h_old_to_new[h_id_old]
            x_val = float(h_kmeans.cluster_centers_[cluster_id][0])
            y_val = float(h_kmeans.cluster_centers_[cluster_id][1])
            locations.append({"index": h_id, "type": "H", "x": x_val, "y": y_val})
            h_id_to_xy[h_id] = (x_val, y_val)

        # Place L/E farther from Hubs based on configured strategy.
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
            h_weights = h_filtered['daily_count'].to_numpy(dtype=float)
            h_total_w = max(float(h_weights.sum()), 1.0)
            l_cx = float((h_filtered['x_km'] * h_weights).sum() / h_total_w)
            l_cy = float((h_filtered['y_km'] * h_weights).sum() / h_total_w)

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
            hub_hotspots = list(h_id_to_xy.values())
            for i in range(num_bases):
                if i < len(hub_hotspots):
                    x_val, y_val = hub_hotspots[i]
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
        hub_hotspots = list(h_id_to_xy.values())
        h_weights = h_filtered['daily_count'].to_numpy(dtype=float)
        h_total_w = max(float(h_weights.sum()), 1.0)
        e_cx = float((h_filtered['x_km'] * h_weights).sum() / h_total_w)
        e_cy = float((h_filtered['y_km'] * h_weights).sum() / h_total_w)
        ring_radius_e = min(max(spread_km, 0.5), map_half_size_km * 0.75)

        for i in range(num_bases + 1, num_swapping + 1):
            idx = i - (num_bases + 1)
            if placement_mode == "centroid":
                theta = (2.0 * np.pi * idx) / max(num_swapping - num_bases, 1)
                x_val = e_cx + ring_radius_e * np.cos(theta)
                y_val = e_cy + ring_radius_e * np.sin(theta)
            elif idx < len(hub_hotspots):
                x_val, y_val = hub_hotspots[idx]
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
        h_coords = set(zip(loc_df[loc_df['type'] == 'H']['x'].round(6), loc_df[loc_df['type'] == 'H']['y'].round(6)))
        e_coords = set(zip(loc_df[loc_df['type'] == 'E']['x'].round(6), loc_df[loc_df['type'] == 'E']['y'].round(6)))
        overlap_he = h_coords & e_coords

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
            f"  aggregation mode: clustered to targets (H={num_hubs})\n"
            f"  L/E placement mode: {placement_mode}\n"
            f"  work-hour window: {work_hour_start}:00-{work_hour_end}:59 ({work_hours_count}h), tz={config.MEITUAN_TIMEZONE}\n"
            f"  map half-size: {map_half_size_km:.2f} km (side {2 * map_half_size_km:.2f} km)\n"
            f"  raw(date) orders={len(df)}, unique origins={total_unique_p_day}, unique destinations={total_unique_d_day}, unique OD pairs={total_pairs_day}\n"
            f"  work-hour orders={len(work_df)}, unique origins={total_unique_p_work}, unique destinations={total_unique_d_work}, unique OD pairs={total_pairs_work}\n"
            f"  in-map orders={len(area_df)}, unique origins={area_unique_p}, unique destinations={area_unique_d}, unique OD pairs={area_pairs}\n"
            f"  after node frequency filter >= {min_node_hourly_orders:.3f}/h: H candidates={len(h_filtered)}\n"
            f"  clustered nodes before OD pruning: H={n_h}; after OD pruning: H={len(h_id_to_xy)}\n"
            f"  selected nodes: H={len(h_id_to_xy)}, L={num_bases}, E={num_swapping}\n"
            f"  retained OD pairs: {len(orders_out)}, total clustered demand(per-hour)={orders_out['demand'].sum():.4f}\n"
            f"  min E-E target distance={min_ee_distance_km:.2f} km\n"
            f"  H-E coordinate overlap count={len(overlap_he)}"
        )

        if overlap_he:
            examples = list(overlap_he)[:5]
            print(f"  overlap examples (x_km,y_km): {examples}")
            print(f"  note: E1..E{num_bases} are intentionally colocated with L. H-E overlap above is unintended coincidence/data geometry.")
