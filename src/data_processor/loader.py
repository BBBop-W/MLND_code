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
                
                inst.coords[node_id] = (x, y)
                
                if node_type == "L":
                    inst.L_hat.append(node_id)
                    inst.f[node_id] = config.COST_BASE
                elif node_type == "P":
                    inst.P.append(node_id)
                    inst.f[node_id] = config.COST_PICKUP
                    inst.mu[node_id] = config.MU_PICKUP
                elif node_type == "D":
                    inst.D.append(node_id)
                    inst.f[node_id] = config.COST_DELIVERY
                    inst.mu[node_id] = config.MU_DELIVERY
                elif node_type == "E":
                    inst.E.append(node_id)
                    inst.f[node_id] = config.COST_SWAPPING
                    inst.mu[node_id] = config.MU_SWAPPING
                    
        # 2. Virtual Bases (linked to delivery nodes)
        for d in inst.D:
            v_id = f"V_{d}"
            inst.L_v.append(v_id)
            inst.coords[v_id] = inst.coords[d] # Same location
            
        # 3. Compute Distances and Times
        all_physical_nodes = inst.L_hat + inst.P + inst.D + inst.E
        for u in all_physical_nodes:
            for v in all_physical_nodes:
                dist = np.sqrt((inst.coords[u][0] - inst.coords[v][0])**2 + 
                               (inst.coords[u][1] - inst.coords[v][1])**2)
                inst.dist_matrix[(u, v)] = dist
                inst.time_matrix[(u, v)] = dist / 1.0 # speed = 1.0
                
        # 4. Read orders
        with open(orders_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row["start_node"]
                d = row["end_node"]
                demand = float(row["demand"])
                inst.h[(p, d)] = demand
                inst.A_PD.add((p, d))
                
        # 5. Parse valid arcs and structures for Gurobi
        inst.parse_for_gurobi()
        
        return inst
