import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
import pickle
from src import config
from src.data_processor.instance import Instance

class ResultProcessor:
    def __init__(self, model, instance: Instance):
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
            
        try:
            self.gamma = self.model.getVarByName("Gamma").X
        except:
            self.gamma = None
        
        self.x = {}
        self.y = {}
        self.u = {}
        self.eta = {}
        self.xi = {}
        self.psi = {}
        self.phi = {}
        self.z = {}
        self.rho = {}
        self.lambd = {}
        self.theta = {}

        Vars = self.model.getVars()
        
        for v in Vars:
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
            elif v.VarName.startswith("eta["):
                parts = v.VarName[4:-1].split(',')
                if len(parts) == 2:
                    self.u[(parts[0], int(parts[1]))] = v.X
            elif v.VarName.startswith("xi["):
                key = v.VarName[3:-1]
                self.xi[key] = v.X
            elif v.VarName.startswith("psi["):
                parts = v.VarName[4:-1].split(',')
                if len(parts) == 2:
                    self.psi[(parts[0], parts[1])] = v.X
            elif v.VarName.startswith("phi["):
                parts = v.VarName[4:-1].split(',')
                if len(parts) == 2:
                    self.phi[(parts[0], parts[1])] = v.X
            elif v.VarName.startswith("z["):
                parts = v.VarName[2:-1].split(',')
                if len(parts) == 2:
                    self.z[(parts[0], int(parts[1]))] = v.X
            elif v.VarName.startswith("rho_l["):
                key = v.VarName[6:-1]
                self.rho[key] = v.X
            elif v.VarName.startswith("rho_ell["):
                key = v.VarName[8:-1]
                self.rho[key] = v.X
            elif v.VarName.startswith("lambda["):
                key = v.VarName[7:-1]
                self.lambd[key] = v.X
            elif v.VarName.startswith("theta["):
                parts = v.VarName[6:-1].split(',')
                if len(parts) == 3:
                    self.theta[(parts[0], parts[1], parts[2])] = v.X

        self.open_L = sum(1 for node in self.inst.L if self.x.get(node, 0) > 0.5)
        self.open_H = sum(1 for node in self.inst.H if self.x.get(node, 0) > 0.5)
        self.open_E = sum(1 for node in self.inst.E if self.x.get(node, 0) > 0.5)
        # Keep explicit alias for downstream consumers after eta rename.
        self.eta = dict(self.u)

        # Summary nodes include physical nodes plus virtual bases L_v.
        self.summary_nodes = list(self.inst.L) + list(self.inst.L_v) + list(self.inst.H) + list(self.inst.E)

        # Node-level flow accounting for printed/exported summaries.
        # Prefer theta-based leg decomposition so swapping stations (E)
        # receive correct in/out flow. Fallback to psi/phi when theta is absent.
        self.node_inflow = {node: 0.0 for node in self.summary_nodes}
        self.node_outflow = {node: 0.0 for node in self.summary_nodes}

        flow_tol = 1e-5

        def add_leg(u_node, v_node, flow):
            if u_node in self.node_outflow:
                self.node_outflow[u_node] += flow
            if v_node in self.node_inflow:
                self.node_inflow[v_node] += flow

        has_theta_flow = any(flow > flow_tol for flow in self.theta.values())
        if has_theta_flow:
            for (u_node, v_node, k_node), flow in self.theta.items():
                if flow <= flow_tol:
                    continue
                if k_node == '0':
                    add_leg(u_node, v_node, flow)
                else:
                    # Swapped flow is decomposed into two flown legs.
                    add_leg(u_node, k_node, flow)
                    add_leg(k_node, v_node, flow)
        else:
            for (u_node, v_node), flow in self.psi.items():
                if flow <= flow_tol:
                    continue
                add_leg(u_node, v_node, flow)
            for (u_node, v_node), flow in self.phi.items():
                if flow <= flow_tol:
                    continue
                add_leg(u_node, v_node, flow)

        # Validate consistency: if assignment implies non-zero base throughput,
        # physical base flow must not be zero.
        hub_outgoing = {h_node: 0.0 for h_node in self.inst.H}
        for (i_node, j_node), flow in self.psi.items():
            if i_node in hub_outgoing and flow > flow_tol:
                hub_outgoing[i_node] += flow

        self.base_assigned_flow = {l_node: 0.0 for l_node in self.inst.L}
        for (i_node, l_node), y_val in self.y.items():
            if y_val <= flow_tol:
                continue
            if l_node in self.base_assigned_flow and i_node in hub_outgoing:
                self.base_assigned_flow[l_node] += y_val * hub_outgoing[i_node]

        inconsistent_bases = []
        for l_node in self.inst.L:
            expected_flow = self.base_assigned_flow.get(l_node, 0.0)
            if expected_flow <= flow_tol:
                continue
            phys_in = self.node_inflow.get(l_node, 0.0)
            phys_out = self.node_outflow.get(l_node, 0.0)
            if phys_in <= flow_tol and phys_out <= flow_tol:
                inconsistent_bases.append((l_node, expected_flow, phys_in, phys_out))

        if inconsistent_bases:
            preview = "; ".join(
                f"{l}: expected~{exp_f:.4f}, in={in_f:.4f}, out={out_f:.4f}"
                for l, exp_f, in_f, out_f in inconsistent_bases[:5]
            )
            raise ValueError(
                "Model-flow inconsistency detected: base has assignment-implied throughput "
                "but zero physical flow. "
                f"Examples: {preview}"
            )

        self.node_total_flow = {
            node: self.node_inflow[node] + self.node_outflow[node]
            for node in self.summary_nodes
        }

        # Use a flow-based served-demand metric so summaries remain stable even if
        # objective reporting changes (e.g., multi-objective/tie-break settings).
        self.served_demand = sum(
            self.psi.get((i, j), 0.0) for (i, j) in self.inst.A_HH
        )

        # Profit/cost accounting used by summary output.
        self.served_revenue = config.ORDER_REVENUE_COEF * self.served_demand
        self.fixed_cost = sum(
            self.inst.f.get(node, 0.0) * self.x.get(node, 0.0) for node in self.inst.L + self.inst.E
        )
        f_hub = config.COST_HUB
        self.fixed_cost += sum(f_hub[c] * self.z.get((i, c), 0.0) for i in self.inst.H for c in range(len(config.MU_HUB)))
        flight_coef = getattr(config, "COST_FLIGHT_TIME", 0.1)
        self.flight_cost = flight_coef * sum(v for v in self.theta.values())
        gamma_val = self.gamma if self.gamma is not None else 0.0
        self.drone_cost = config.COST_DRONE * gamma_val
        self.total_cost = self.fixed_cost + self.flight_cost + self.drone_cost
        self.net_profit = self.served_revenue - self.total_cost
        self.rho_avg = float(np.mean(list(self.rho.values()))) if self.rho else 0.0



    def _rho_to_l_metric(self, rho_val):
        if rho_val is None:
            return None
        tol = 1e-5
        if rho_val >= 1.0 - tol:
            return None
        return rho_val / (1.0 - rho_val)


    def _is_all_zero_row(self, open_raw, in_flow, out_flow, rho_val, l_val):
        tol = 1e-5
        rho_num = 0.0 if rho_val is None else float(rho_val)
        l_num = 0.0 if l_val is None else float(l_val)
        open_is_zero = (open_raw is not None) and (abs(float(open_raw)) <= tol)
        return (
            abs(in_flow) <= tol
            and abs(out_flow) <= tol
            and abs(rho_num) <= tol
            and abs(l_num) <= tol
            and open_is_zero
        )

    def print_summary(self):
        print("\n" + "="*50)
        if self.obj_val is not None:
            print(f"OPTIMIZATION RESULT SUMMARY: {self.obj_val:.4f}")
        else:
            print("OPTIMIZATION RESULT SUMMARY")
        print("="*50)
        print(f"Net Profit                   : {self.net_profit:.4f}")
        print(f"Completed Volume             : {self.served_demand:.4f}")
        print(f"Completed Volume Revenue     : {self.served_revenue:.4f}")
        print(f"Total Cost                   : {self.total_cost:.4f}")
        print(f"  - Fixed Station Cost       : {self.fixed_cost:.4f}")
        print(f"  - Flight Time Cost         : {self.flight_cost:.4f}")
        print(f"  - Drone Cost               : {self.drone_cost:.4f}")
        print(
            f"Model Objective              : {self.obj_val:.4f}"
            if self.obj_val is not None
            else "Model Objective              : None"
        )
        if self.gamma is not None:
            print(f"Total Aircraft (gamma)        : {self.gamma:.4f}")
        else:
            print("Total Aircraft (gamma)        : N/A")
        print(f"Average Service Level (rho)  : {self.rho_avg:.4f}" if self.rho else "Average Service Level (rho)  : N/A")
        print(f"Runtime                      : {self.run_time:.2f}s")
        print(f"Gap                          : {self.gap*100:.2f}%" if self.gap is not None else "Gap: N/A")
        
        print(f"\nOpen Stations:")
        print(f"  Bases (L)       : {self.open_L} / {len(self.inst.L)}")
        print(f"  Hubs (H)        : {self.open_H} / {len(self.inst.H)}")
        print(f"  Swapping (E)    : {self.open_E} / {len(self.inst.E)}")
        
        print(f"\nActive Assignments (y_li > 0.01):")
        y_active = [(i, l, val) for (i, l), val in self.y.items() if val > 0.01]

        for i, l, val in y_active[:10]:
            rho_l = self.rho.get(l, 0.0)
            print(f"  Base {l} assigned to Hub {i} with y={val:.4f}, rho_l={rho_l:.4f}")
        if len(y_active) > 10:
            print(f"  ... and {len(y_active) - 10} more.")

        flow_tol = 1e-5
        active_phi_edges = sum(1 for v in self.phi.values() if v > flow_tol)
        active_psi_edges = sum(1 for v in self.psi.values() if v > flow_tol)
        active_theta_edges = sum(1 for v in self.theta.values() if v > flow_tol)
        print(f"\nActive edge counts (tol={flow_tol:g}):")
        print(f"  phi edges       : {active_phi_edges}")
        print(f"  psi edges       : {active_psi_edges}")
        print(f"  theta edges     : {active_theta_edges}")

        # Dangling open stations: open physical nodes with no significant incident flow.
        dangling = []
        for node in self.inst.N:
            if self.x.get(node, 0) <= 0.5:
                continue
            incident = self.node_inflow.get(node, 0.0) + self.node_outflow.get(node, 0.0)
            if incident <= flow_tol:
                dangling.append(node)
        if dangling:
            print(f"Warning: {len(dangling)} open stations have no incident flow above tolerance.")
            preview = ", ".join(dangling[:10])
            print(f"  Examples: {preview}{' ...' if len(dangling) > 10 else ''}")

        print("\nPer-node flow summary:")
        print("  NodeID | Open | InFlow | OutFlow | Lambda |   rho |     L |    Xi")
        for node in self.summary_nodes:
            if node in self.inst.L_v:
                open_raw = self.x.get(self.inst.lv_to_h.get(node), None)
            else:
                open_raw = self.x.get(node, None)
            open_display = str(int(open_raw > 0.5)) if open_raw is not None else "-"
            in_flow = self.node_inflow.get(node, 0.0)
            out_flow = self.node_outflow.get(node, 0.0)
            lambda_val = self.lambd.get(node, None)
            rho_val = self.rho.get(node, None)
            l_val = self._rho_to_l_metric(rho_val)
            if self._is_all_zero_row(open_raw, in_flow, out_flow, rho_val, l_val):
                continue
            lambda_display = f"{lambda_val:6.3f}" if lambda_val is not None else "  -   "
            rho_display = f"{rho_val:5.3f}" if rho_val is not None else "  -  "
            l_display = f"{l_val:5.3f}" if l_val is not None else "  -  "
            xi_val = self.xi.get(node, None)
            xi_display = f"{xi_val:5.3f}" if xi_val is not None else "  -  "
            print(f"  {node:>6} | {open_display:>4} | {in_flow:7.3f} | {out_flow:8.3f} | {lambda_display:>6} | {rho_display:>5} | {l_display:>5} | {xi_display:>5}")

                
        print("="*50 + "\n")

    def save_node_flow_summary(self, save_path):
        """Export CSV of node-level flow summary with rho and L columns."""
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NodeID', 'Open', 'InFlow', 'OutFlow', 'Lambda', 'rho', 'L', 'Xi'])
            for node in self.summary_nodes:
                rho_val = self.rho.get(node, None)
                l_val = self._rho_to_l_metric(rho_val)
                in_flow = self.node_inflow.get(node, 0.0)
                out_flow = self.node_outflow.get(node, 0.0)
                lambda_val = self.lambd.get(node, None)
                xi_val = self.xi.get(node, None)
                if node in self.inst.L_v:
                    open_raw = self.x.get(self.inst.lv_to_h.get(node), None)
                else:
                    open_raw = self.x.get(node, None)
                if self._is_all_zero_row(open_raw, in_flow, out_flow, rho_val, l_val):
                    continue
                writer.writerow([
                    node,
                    1 if (open_raw is not None and open_raw > 0.5) else (0 if open_raw is not None else ''),
                    in_flow,
                    out_flow,
                    lambda_val,
                    rho_val,
                    l_val,
                    xi_val,
                ])

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
                'z': self.z,
                'u': self.u,
                'eta': self.eta,
                'phi': self.phi,
                'psi': self.psi,
                'rho': self.rho,
                'lambda': self.lambd,
                'theta': self.theta,
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
        for node in self.inst.L + self.inst.H + self.inst.E:
            if self.x.get(node, 0) > 0.5:
                x, y = self.inst.coords[node]
                node_type = node[0]
                mu = self.inst.mu.get(node, 0.0)
                rho_val = 0.0
                if node_type == 'L':
                    rho_val = self.rho.get(node, 0.0)
                elif node_type == 'H':
                    chosen_c = None
                    for c in range(len(config.MU_HUB)):
                        if self.z.get((node, c), 0.0) > 0.5:
                            chosen_c = c
                            break
                    if chosen_c is not None:
                        mu = config.MU_HUB[chosen_c]
                    else:
                        mu = 0.0
                    rho_val = (self.lambd.get(node, 0.0) / mu) if mu > 1e-8 else 0.0
                
                rows.append({
                    'index': node,
                    'type': node_type,
                    'x': x,
                    'y': y,
                    'mu': mu,
                    'rho': rho_val
                })
        df = pd.DataFrame(rows, columns=['index', 'type', 'x', 'y', 'mu', 'rho'])
        df.to_csv(save_path, index=False)
        print(f"Open facilities config saved to {save_path}")

    def plot_network(self, save_path=None):
        G = nx.DiGraph()
        flow_tol = 1e-5

        def is_corresponding_base_swap(a_node, b_node):
            if not ((a_node.startswith('L') and b_node.startswith('E')) or (a_node.startswith('E') and b_node.startswith('L'))):
                return False
            try:
                return int(a_node[1:]) == int(b_node[1:])
            except ValueError:
                return False
        
        # Add all physical nodes
        for node in self.inst.N:
            if node in self.inst.L_v:
                status = 'open' if self.x.get(self.inst.lv_to_h.get(node), 0) > 0.5 else 'closed'
            else:
                status = 'open' if self.x.get(node, 0) > 0.5 else 'closed'
            node_type = node[0] 
            G.add_node(node, pos=self.inst.coords[node], status=status, type=node_type)
            
        plt.figure(figsize=(16, 14))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Calculate max flow for scaling
        all_flows = [f for f in self.theta.values() if f > flow_tol]
        max_phi_flow = max(all_flows) if all_flows else 1.0
        
        # 1. Plot direct flows (phi^0) - Gray lines
        direct_edges = []
        direct_widths = []
        for (u, v, k), flow in self.theta.items():
            if k == '0' and flow > flow_tol:
                u_plot = self.inst.lv_to_h.get(u, u)
                v_plot = self.inst.lv_to_h.get(v, v)
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
        for (u, v, k), flow in self.theta.items():
            if k != '0' and flow > flow_tol:
                u_plot = self.inst.lv_to_h.get(u, u)
                v_plot = self.inst.lv_to_h.get(v, v)

                # Leg 1: origin to swap
                if u_plot != k and not is_corresponding_base_swap(u_plot, k):
                    swap_edges.append((u_plot, k))
                # Leg 2: swap to dest
                if k != v_plot and not is_corresponding_base_swap(k, v_plot):
                    swap_edges.append((k, v_plot))
                
                width = 1.0 + 4.0 * (flow / max_phi_flow)
                if u_plot != k and not is_corresponding_base_swap(u_plot, k):
                    swap_widths.append(width)
                if k != v_plot and not is_corresponding_base_swap(k, v_plot):
                    swap_widths.append(width)

        if swap_edges:
            nx.draw_networkx_edges(G, pos, edgelist=swap_edges, width=swap_widths, 
                                edge_color='darkorange', alpha=0.7, arrows=True, 
                                arrowsize=15, connectionstyle='arc3,rad=0.1',
                                label='Swapped Flow (phi^k)')

        # If phi edges are unavailable (e.g., master-only model), fall back to psi.
        if not direct_edges and not swap_edges:
            psi_edges = []
            psi_widths = []
            max_psi_flow = max([f for f in self.psi.values() if f > flow_tol], default=1.0)
            for (u, v), flow in self.psi.items():
                if flow <= flow_tol:
                    continue
                u_plot = self.inst.lv_to_h.get(u, u)
                v_plot = self.inst.lv_to_h.get(v, v)
                if u_plot != v_plot:
                    psi_edges.append((u_plot, v_plot))
                    psi_widths.append(1.0 + 4.0 * (flow / max_psi_flow))
            if psi_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=psi_edges,
                    width=psi_widths,
                    edge_color='steelblue',
                    alpha=0.6,
                    arrows=True,
                    arrowsize=15,
                    connectionstyle='arc3,rad=0.1',
                    label='Aggregated Flow (psi)',
                )
            else:
                print("Warning: no flow edges above tolerance to plot.")

        # 3. Nodes - distinguish open and closed
        shapes = {'L': 's', 'H': '^', 'E': 'D', 'V': 'o'}
        colors_open = {'L': 'red', 'H': 'green', 'E': 'orange', 'V': 'steelblue'}
        
        for n_type in ['L', 'H', 'E', 'V']:
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

        # Label every node ID next to its marker for easier auditing.
        for node, (x_coord, y_coord) in pos.items():
            plt.text(
                x_coord + 0.03,
                y_coord + 0.03,
                node,
                fontsize=8,
                color='black',
                alpha=0.85,
            )

        x_vals = [coord[0] for coord in pos.values()]
        y_vals = [coord[1] for coord in pos.values()]
        if x_vals and y_vals:
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)
            x_span = x_max - x_min
            y_span = y_max - y_min
            y_arrow = y_min - 0.08 * (y_span if y_span > 0 else 1.0)
            plt.annotate(
                "",
                xy=(x_max, y_arrow),
                xytext=(x_min, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2),
            )
            plt.text(
                (x_min + x_max) / 2.0,
                y_arrow - 0.03 * (y_span if y_span > 0 else 1.0),
                f"Left-right span: {x_span:.2f} km",
                ha='center',
                va='top',
                fontsize=10,
                color='black',
            )

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
