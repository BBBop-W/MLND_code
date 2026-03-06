import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
import pickle

class ResultProcessor:
    def __init__(self, model, instance):
        self.model = model.model
        self.inst = instance
        
        if self.model.SolCount == 0:
            raise ValueError("No solution found in the model.")
            
        # Try to get objVal and MIPGap safely
        try:
            self.obj_val = self.model.objVal
        except:
            self.obj_val = None
            
        try:
            self.gap = self.model.MIPGap
        except:
            self.gap = None
            
        self.run_time = self.model.Runtime
            
        self.gamma = self.model.getVarByName("gamma").X
        
        self.x = {}
        self.y = {}
        self.u = {}
        self.psi = {}
        self.phi = {}
        
        for v in self.model.getVars():
            if v.VarName.startswith("x["):
                key = v.VarName[2:-1]
                self.x[key] = v.X
            elif v.VarName.startswith("y["):
                parts = v.VarName[2:-1].split(',')
                if len(parts) == 2:
                    self.y[(parts[0], parts[1])] = v.X
            elif v.VarName.startswith("u["):
                parts = v.VarName[2:-1].split(',')
                if len(parts) == 2:
                    self.u[(parts[0], int(parts[1]))] = v.X
            elif v.VarName.startswith("psi["):
                parts = v.VarName[4:-1].split(',')
                if len(parts) == 2:
                    self.psi[(parts[0], parts[1])] = v.X
            elif v.VarName.startswith("phi["):
                parts = v.VarName[4:-1].split(',')
                if len(parts) == 3:
                    u_node = parts[0]
                    v_node = parts[1]
                    k_node = parts[2]
                    self.phi[(u_node, v_node, k_node)] = v.X
                    
        self.open_L = sum(1 for node in self.inst.L_hat if self.x.get(node, 0) > 0.5)
        self.open_P = sum(1 for node in self.inst.P if self.x.get(node, 0) > 0.5)
        self.open_D = sum(1 for node in self.inst.D if self.x.get(node, 0) > 0.5)
        self.open_E = sum(1 for node in self.inst.E if self.x.get(node, 0) > 0.5)
        
        from src import config
        active_rhos = [config.SERVICE_LEVELS[s] for l in self.inst.L_hat for s in range(len(config.SERVICE_LEVELS)) if self.u.get((l, s), 0) > 0.5]
        self.rho = sum(active_rhos) / len(active_rhos) if active_rhos else 0.0

    def print_summary(self):
        print("\n" + "="*50)
        print("OPTIMIZATION RESULT SUMMARY")
        print("="*50)
        print(f"Objective Value (Served Demand): {self.obj_val:.4f}" if self.obj_val else "Objective Value: None")
        print(f"Average Service Level (rho)  : {self.rho:.4f}")
        print(f"Total Fleet Size (gamma)     : {self.gamma:.4f}")
        print(f"Runtime                      : {self.run_time:.2f}s")
        print(f"Gap                          : {self.gap*100:.2f}%" if self.gap is not None else "Gap: N/A")
        
        print(f"\nOpen Stations:")
        print(f"  Bases (L)       : {self.open_L} / {len(self.inst.L_hat)}")
        print(f"  Pickups (P)     : {self.open_P} / {len(self.inst.P)}")
        print(f"  Deliveries (D)  : {self.open_D} / {len(self.inst.D)}")
        print(f"  Swapping (E)    : {self.open_E} / {len(self.inst.E)}")
        
        print(f"\nActive Assignments (y_li > 0.01):")
        y_active = [(l, p, val) for (l, p), val in self.y.items() if val > 0.01]

        from src import config
        for l, p, val in y_active[:10]:
            rho = [config.SERVICE_LEVELS[s] for s in range(len(config.SERVICE_LEVELS)) if  self.u.get((l, s), 0)> 0.01]
            print(f"  Base {l} assigned to Pickup {p} with weight {val:.4f} and rho {rho}")
        if len(y_active) > 10:
            print(f"  ... and {len(y_active) - 10} more.")
        print("="*50 + "\n")

    def save_selected_stations(self, save_path):
        """Export CSV of selected (open) stations."""
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NodeID', 'Type', 'x', 'y'])
            for node in self.inst.N:
                if self.x.get(node, 0) > 0.5:
                    x, y = self.inst.coords[node]
                    node_type = node[0]
                    writer.writerow([node, node_type, x, y])

    def save_python_object(self, save_path):
        """Save instance and parsed results natively as a pickle file."""
        data = {
            'instance': self.inst,
            'variables': {
                'x': self.x,
                'y': self.y,
                'u': self.u,
                'phi': self.phi,
                'psi': self.psi,
                'rho': self.rho,
                'gamma': self.gamma
            },
            'metrics': {
                'obj_val': self.obj_val,
                'run_time': self.run_time,
                'gap': self.gap
            }
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def save_open_facilities_config(self, save_path):
        """Export CSV of open facilities and their parameters for simulation."""
        import pandas as pd
        from src import config

        rows = []
        for node in self.inst.N:
            if self.x.get(node, 0) > 0.5:
                x, y = self.inst.coords[node]
                node_type = node[0]
                mu = self.inst.mu.get(node, 0)
                # For bases, find the selected service level
                rho_val = 0.0
                if node_type == 'L':
                    for s in range(len(config.SERVICE_LEVELS)):
                        if self.u.get((node, s), 0) > 0.5:
                            rho_val = config.SERVICE_LEVELS[s]
                            break
                
                rows.append({
                    'index': node,
                    'type': node_type,
                    'x': x,
                    'y': y,
                    'mu': mu,
                    'rho': rho_val
                })
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"Open facilities config saved to {save_path}")

    def plot_network(self, save_path=None):
        G = nx.DiGraph()
        
        # Add all physical nodes
        for node in self.inst.N:
            status = 'open' if self.x.get(node, 0) > 0.5 else 'closed'
            node_type = node[0] 
            G.add_node(node, pos=self.inst.coords[node], status=status, type=node_type)
            
        plt.figure(figsize=(16, 14))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Calculate max flow for scaling
        all_flows = [f for f in self.phi.values() if f > 0.01]
        max_phi_flow = max(all_flows) if all_flows else 1.0
        
        # 1. Plot direct flows (phi^0) - Gray lines
        direct_edges = []
        direct_widths = []
        for (u, v, k), flow in self.phi.items():
            if k == '0' and flow > 0.01:
                u_plot = self.inst.lv_to_d.get(u, u)
                v_plot = self.inst.lv_to_d.get(v, v)
                if u_plot != v_plot:
                    direct_edges.append((u_plot, v_plot))
                    # Visibly thicker lines
                    direct_widths.append(1.0 + 4.0 * (flow / max_phi_flow))

        if direct_edges:
            nx.draw_networkx_edges(G, pos, edgelist=direct_edges, width=direct_widths, 
                                   edge_color='gray', alpha=0.4, arrows=True, 
                                   arrowsize=15, connectionstyle='arc3,rad=0.1', 
                                   label='Direct Flow (phi^0)')

        # 2. Plot flows via swapping stations (phi^k) - Orange lines
        swap_edges = []
        swap_widths = []
        for (u, v, k), flow in self.phi.items():
            if k != '0' and flow > 0.01:
                u_plot = self.inst.lv_to_d.get(u, u)
                v_plot = self.inst.lv_to_d.get(v, v)
                
                # Leg 1: origin to swap
                swap_edges.append((u_plot, k))
                # Leg 2: swap to dest
                swap_edges.append((k, v_plot))
                
                width = 1.0 + 4.0 * (flow / max_phi_flow)
                swap_widths.append(width)
                swap_widths.append(width)

        if swap_edges:
            nx.draw_networkx_edges(G, pos, edgelist=swap_edges, width=swap_widths, 
                                   edge_color='darkorange', alpha=0.7, arrows=True, 
                                   arrowsize=15, connectionstyle='arc3,rad=0.1',
                                   label='Swapped Flow (phi^k)')

        # 3. Nodes - distinguish open and closed
        shapes = {'L': 's', 'P': '^', 'D': 'o', 'E': 'D'}
        colors_open = {'L': 'red', 'P': 'green', 'D': 'blue', 'E': 'orange'}
        
        for n_type in ['L', 'P', 'D', 'E']:
            open_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'open' and attr['type'] == n_type]
            if open_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=open_nodes, 
                                       node_color=colors_open[n_type], 
                                       node_shape=shapes[n_type], 
                                       node_size=300, 
                                       label=f'Open {n_type}')
                                       
            closed_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'closed' and attr['type'] == n_type]
            if closed_nodes:
                # Hollow shapes for closed nodes
                nx.draw_networkx_nodes(G, pos, nodelist=closed_nodes, 
                                       node_color='white', 
                                       edgecolors=colors_open[n_type],
                                       node_shape=shapes[n_type], 
                                       node_size=100, alpha=0.3,
                                       linewidths=1.5,
                                       label=f'Closed {n_type}')

        plt.title('Optimized Drone Delivery Network (Direct vs Swapped)', fontsize=16)
        plt.xlabel('X Coordinate (km)', fontsize=12)
        plt.ylabel('Y Coordinate (km)', fontsize=12)
        
        # Cleanup legend - avoid duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l not in by_label:
                by_label[l] = h
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium', ncol=1)
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to {save_path}")
        else:
            plt.show()
