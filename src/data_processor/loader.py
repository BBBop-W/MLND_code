import csv
import numpy as np
from .instance import Instance
from .. import config

class DataLoader:
    def __init__(self):
        pass

    def load_instance(self, locations_path: str, orders_path: str) -> Instance:
        """
        Reads locations.csv and orders.csv to populate and return an Instance object.
        """
        inst = Instance()
        
        # 1. Read locations
        with open(locations_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row["index"]
                node_type = row["type"]
                x = float(row["x"])
                y = float(row["y"])

                if not np.isfinite(x) or not np.isfinite(y):
                    raise ValueError(f"Invalid coordinate for node {node_id}: ({x}, {y})")
                
                inst.coords[node_id] = (x, y)
                
                if node_type == "L":
                    inst.L.append(node_id)
                    inst.f[node_id] = config.COST_BASE
                elif node_type == "H" or node_type == "P" or node_type == "D":
                    inst.H.append(node_id)
                    # Hub cost/service level is selected by binary z[i,c] in the model,
                    # so keep scalar placeholders here to avoid list-valued node attributes.
                    inst.f[node_id] = 0.0
                    inst.mu[node_id] = 0.0
                
                elif node_type == "E":
                    inst.E.append(node_id)
                    inst.f[node_id] = config.COST_SWAPPING
                    inst.mu[node_id] = config.MU_SWAPPING
                    
        # 2. Virtual Bases (linked to delivery nodes)
        for h in inst.H:
            v_id = f"V_{h}"
            inst.L_v.append(v_id)
            inst.coords[v_id] = inst.coords[h]
            
        # 3. Compute Distances and Times
        if config.DRONE_SPEED_KMPH <= 0:
            raise ValueError("DRONE_SPEED_KMPH must be positive for consistent time conversion.")

        all_physical_nodes = inst.L + inst.H + inst.L_v + inst.E
        for u in all_physical_nodes:
            for v in all_physical_nodes:
                dist = np.sqrt((inst.coords[u][0] - inst.coords[v][0])**2 + 
                            (inst.coords[u][1] - inst.coords[v][1])**2)
                inst.dist_matrix[(u, v)] = dist
                # Distance in km, time in hour.
                inst.time_matrix[(u, v)] = dist / config.DRONE_SPEED_KMPH
                
        # 4. Read orders
        with open(orders_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row["start_node"]
                d = row["end_node"]
                demand = float(row["demand"])

                if demand < 0:
                    raise ValueError(f"Negative demand found for ({p}, {d}): {demand}")

                inst.h[(p, d)] = demand
                inst.A_HH.add((p, d))
                
        # 5. Parse valid arcs and structures for Gurobi
        inst.parse_for_gurobi()
        
        return inst
