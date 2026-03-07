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
        
        locations_path = os.path.join(save_dir, "locations.csv")
        orders_path = os.path.join(save_dir, "orders.csv")
        
        locations = []
        for i in range(1, config.NUM_BASES + 1):
            locations.append({"index": f"L{i}", "type": "L", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
        for i in range(1, config.NUM_PICKUP + 1):
            locations.append({"index": f"P{i}", "type": "P", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
        for i in range(1, config.NUM_DELIVERY + 1):
            locations.append({"index": f"D{i}", "type": "D", "x": np.random.uniform(0, config.AREA_SIZE), "y": np.random.uniform(0, config.AREA_SIZE)})
        for i in range(1, config.NUM_SWAPPING + 1):
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
                                date_filter: int = 20221017):
        import pandas as pd
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Reading data from {raw_data_path} for date {date_filter}...")
        df = pd.read_csv(raw_data_path)
        df = df[df['dt'] == date_filter].copy()
        
        if df.empty:
            raise ValueError(f"No data found for date {date_filter}")
        
        top_poi = df['poi_id'].value_counts().idxmax()
        center_node = df[df['poi_id'] == top_poi].iloc[0]
        c_lng, c_lat = center_node['sender_lng'], center_node['sender_lat']
        delta = 60000 
        
        area_df = df[
            (df['sender_lng'] >= c_lng - delta) & (df['sender_lng'] <= c_lng + delta) &
            (df['sender_lat'] >= c_lat - delta) & (df['sender_lat'] <= c_lat + delta)
        ].copy()
        
        area_df['raw_dist'] = np.sqrt((area_df['sender_lng'] - area_df['recipient_lng'])**2 + 
                                      (area_df['sender_lat'] - area_df['recipient_lat'])**2)
        min_dist_threshold = 25000 
        separated_df = area_df[area_df['raw_dist'] >= min_dist_threshold].copy()
        
        if len(separated_df) < 50:
            print("Warning: Too few long-distance orders. Relaxing distance threshold.")
            separated_df = area_df[area_df['raw_dist'] >= 10000].copy()

        print(f"Area orders: {len(area_df)}. Long-distance orders: {len(separated_df)}")

        # Clustering
        unique_p = separated_df[['sender_lng', 'sender_lat']].drop_duplicates()
        unique_d = separated_df[['recipient_lng', 'recipient_lat']].drop_duplicates()
        n_p = min(num_pickup, len(unique_p))
        n_d = min(num_delivery, len(unique_d))
        
        kmeans_p = KMeans(n_clusters=n_p, random_state=42, n_init="auto")
        p_labels = kmeans_p.fit_predict(unique_p)
        rep_p = pd.DataFrame(kmeans_p.cluster_centers_, columns=['sender_lng', 'sender_lat'])
        p_map = { (row['sender_lng'], row['sender_lat']): (rep_p.loc[p_labels[i], 'sender_lng'], rep_p.loc[p_labels[i], 'sender_lat']) 
                  for i, (idx, row) in enumerate(unique_p.iterrows()) }

        kmeans_d = KMeans(n_clusters=n_d, random_state=42, n_init="auto")
        d_labels = kmeans_d.fit_predict(unique_d)
        rep_d = pd.DataFrame(kmeans_d.cluster_centers_, columns=['recipient_lng', 'recipient_lat'])
        d_map = { (row['recipient_lng'], row['recipient_lat']): (rep_d.loc[d_labels[i], 'recipient_lng'], rep_d.loc[d_labels[i], 'recipient_lat']) 
                  for i, (idx, row) in enumerate(unique_d.iterrows()) }

        # IDs Mapping
        p_id_map = { (row.sender_lng, row.sender_lat): f"P{idx}" for idx, row in enumerate(rep_p.itertuples(), 1) }
        d_id_map = { (row.recipient_lng, row.recipient_lat): f"D{idx}" for idx, row in enumerate(rep_d.itertuples(), 1) }

        # Save simulation events (mapped to node IDs)
        sim_events = separated_df[['platform_order_time', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']].copy()
        def get_p_id(row): return p_id_map[p_map[(row['sender_lng'], row['sender_lat'])]]
        def get_d_id(row): return d_id_map[d_map[(row['recipient_lng'], row['recipient_lat'])]]
        sim_events['p_id'] = sim_events.apply(get_p_id, axis=1)
        sim_events['d_id'] = sim_events.apply(get_d_id, axis=1)
        sim_events.to_csv(os.path.join(save_dir, "simulation_events.csv"), index=False)

        # Projection to KM
        lon_center, lat_center = c_lng / 1e6, c_lat / 1e6
        def to_km_x(lng): return (lng / 1e6 - lon_center) * 111.32 * np.cos(lat_center * np.pi / 180)
        def to_km_y(lat): return (lat / 1e6 - lat_center) * 111.32

        locations = []
        for idx, row in enumerate(rep_p.itertuples(), 1):
            locations.append({"index": f"P{idx}", "type": "P", "x": to_km_x(row.sender_lng), "y": to_km_y(row.sender_lat)})
        for idx, row in enumerate(rep_d.itertuples(), 1):
            locations.append({"index": f"D{idx}", "type": "D", "x": to_km_x(row.recipient_lng), "y": to_km_y(row.recipient_lat)})
        for i in range(1, num_bases + 1):
            locations.append({"index": f"L{i}", "type": "L", "x": np.random.uniform(-6, 6), "y": np.random.uniform(-6, 6)})
        for i in range(1, num_swapping + 1):
            locations.append({"index": f"E{i}", "type": "E", "x": np.random.uniform(-6, 6), "y": np.random.uniform(-6, 6)})
        pd.DataFrame(locations).to_csv(os.path.join(save_dir, "locations.csv"), index=False)

        # Gurobi Input
        raw_orders = separated_df.groupby(['sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']).size().reset_index(name='demand')
        raw_orders['p_id'] = raw_orders.apply(get_p_id, axis=1)
        raw_orders['d_id'] = raw_orders.apply(get_d_id, axis=1)
        cluster_orders = raw_orders.groupby(['p_id', 'd_id'])['demand'].sum().reset_index()
        cluster_orders['demand'] = cluster_orders['demand'] / 24.0
        
        g_orders = []
        for row in cluster_orders.itertuples():
            g_orders.append({"start_node": row.p_id, "end_node": row.d_id, "demand": row.demand})
        pd.DataFrame(g_orders).to_csv(os.path.join(save_dir, "orders.csv"), index=False)
        print(f"Meituan hourly instance generated at: {save_dir}")
