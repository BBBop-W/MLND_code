import gurobipy as gp
from gurobipy import GRB
import numpy as np
from ..data_processor.instance import Instance
from .. import config

class MLNDModel:
    def __init__(self, instance: Instance):
        self.inst = instance
        self.model = gp.Model("MLND_SOCP")
        
        # Accessing sets for convenience
        self.L_hat = self.inst.L_hat # Physical bases (always included)
        
        # Choice of using virtual bases (chaining) for optimization
        if config.USE_VIRTUAL_BASES_OPT:
            self.L_v = self.inst.L_v
            self.L = self.inst.L_hat + self.inst.L_v
        else:
            self.L_v = []
            self.L = self.inst.L_hat
            
        self.P = self.inst.P
        self.D = self.inst.D
        self.E = self.inst.E
        self.N = self.inst.N # Note: N contains physical nodes
        
        # Filter Arcs to only those involving allowed bases
        self.A = [(u, v) for (u, v) in self.inst.A if self._is_arc_allowed(u, v)]
        self.A_PD = [(u, v) for (u, v) in self.inst.A_PD if (u, v) in self.A]
        self.A_LP = [(u, v) for (u, v) in self.inst.A_LP if (u, v) in self.A]
        self.A_DL = [(u, v) for (u, v) in self.inst.A_DL if (u, v) in self.A]
        self.A = self.A_PD + self.A_LP + self.A_DL

        
        self.E_uv = {arc: self.inst.E_uv[arc] for arc in self.A if arc in self.inst.E_uv}
        
        self.S = list(range(len(config.SERVICE_LEVELS)))
        self.rho_vals = config.SERVICE_LEVELS

    def _is_arc_allowed(self, u, v):
        if not config.USE_VIRTUAL_BASES_OPT:
            # Disallow any arc involving a virtual base
            if u.startswith('V_') or v.startswith('V_'):
                return False
        return True

    def build(self):
        # --- Variables ---
        
        # x_u: binary activation for physical nodes and allowed virtual nodes
        x_nodes = self.N + self.L_v
        x = self.model.addVars(x_nodes, vtype=GRB.BINARY, name="x")
        
        # Ensure virtual base activation matches its delivery node
        if config.USE_VIRTUAL_BASES_OPT:
            for d in self.D:
                lv = self.inst.d_to_lv[d]
                self.model.addConstr(x[lv] <= x[d], name=f"x_vbase_{d}")

        # u_ls: binary service level selection
        u = self.model.addVars(self.L, self.S, vtype=GRB.BINARY, name="u")
        
        # y_li: proportion of orders assigned
        y = self.model.addVars(self.A_LP, lb=0.0, ub=1.0, name="y")
        
        # w_lij: linearization of y_li * x_j
        w_keys = [(l, i, j) for (l, i) in self.A_LP for j in self.D if (i, j) in self.A_PD]
        w = self.model.addVars(w_keys, lb=0.0, name="w")
        
        # D_li: aggregated demand
        D_var = self.model.addVars(self.A_LP, lb=0.0, name="D")
        
        # z_lis: linearization of D_li * u_ls
        z_keys = [(l, i, s) for (l, i) in self.A_LP for s in self.S]
        z = self.model.addVars(z_keys, lb=0.0, name="z")
        
        # psi_uv: aggregated physical flow
        psi = self.model.addVars(self.A, lb=0.0, name="psi")
        
        # phi_uv^k: flow decomposition
        phi = {}
        for (u_node, v_node) in self.A:
            dist_uv = self.inst.get_dist(u_node, v_node)
            # Direct flight k='0'
            if dist_uv <= config.MAX_DIRECT_DISTANCE or (u_node in self.D and v_node.startswith('V_')):
                phi[u_node, v_node, '0'] = self.model.addVar(lb=0.0, name=f"phi[{u_node},{v_node},0]")
            for k in self.E_uv.get((u_node, v_node), []):
                phi[u_node, v_node, k] = self.model.addVar(lb=0.0, name=f"phi[{u_node},{v_node},{k}]")

        # SOCP Auxiliaries for Queuing
        xi = self.model.addVars(self.P + self.D, lb=0.0, name="xi") # lambda / (mu - lambda)
        lambd = self.model.addVars(self.P + self.D, lb=0.0, name="lambda")
        
        # Gamma: fleet size
        gamma = self.model.addVar(lb=0.0, name="gamma", vtype=GRB.CONTINUOUS)

        # --- Objective ---
        self.model.setObjective(gp.quicksum(psi[i, j] for (i, j) in self.A_PD), GRB.MAXIMIZE)

        # --- Constraints ---

        # 1. Logical Assignment and Service Level Selection
        for i in self.P:
            l_for_i = [l for l in self.L if (l, i) in self.A_LP]
            if l_for_i:
                self.model.addConstr(gp.quicksum(y[l, i] for l in l_for_i) <= x[i], name=f"assign1_{i}")
        
        for (l, i) in self.A_LP:
            self.model.addConstr(y[l, i] <= x[l], name=f"assign2_{l}_{i}")
            
        for l in self.L:
            self.model.addConstr(gp.quicksum(u[l, s] for s in self.S) == x[l], name=f"service_select_{l}")

        # 2. PWLA Exact Linearization (Big-M)
        for (l, i) in self.A_LP:
            H_i = sum(self.inst.h.get((i, j), 0) for j in self.D if (i, j) in self.A_PD)
            for j in self.D:
                if (i, j) in self.A_PD:
                    self.model.addConstr(w[l, i, j] <= y[l, i], name=f"lin_w1a_{l}_{i}_{j}")
                    self.model.addConstr(w[l, i, j] <= x[j], name=f"lin_w1b_{l}_{i}_{j}")
                    self.model.addConstr(w[l, i, j] >= y[l, i] - (1 - x[j]), name=f"lin_w2_{l}_{i}_{j}")
            
            self.model.addConstr(D_var[l, i] == gp.quicksum(self.inst.h.get((i, j), 0) * w[l, i, j] 
                                                        for j in self.D if (i, j) in self.A_PD), 
                                 name=f"demand_agg_{l}_{i}")
                                 
            for s in self.S:
                self.model.addConstr(z[l, i, s] <= D_var[l, i], name=f"lin_z1a_{l}_{i}_{s}")
                self.model.addConstr(z[l, i, s] <= H_i * u[l, s], name=f"lin_z1b_{l}_{i}_{s}")
                self.model.addConstr(z[l, i, s] >= D_var[l, i] - H_i * (1 - u[l, s]), name=f"lin_z2_{l}_{i}_{s}")
                
            self.model.addConstr(psi[l, i] == gp.quicksum(self.rho_vals[s] * z[l, i, s] for s in self.S), 
                                 name=f"linear_logical_flow_{l}_{i}")

        # 3. Flow Conservation and Decomposition
        all_active_nodes = self.N + self.L_v
        for v in all_active_nodes:
            in_flow = gp.quicksum(psi[u_idx, v_in] for (u_idx, v_in) in self.A if v_in == v)
            out_flow = gp.quicksum(psi[v_out, u_idx] for (v_out, u_idx) in self.A if v_out == v)
            self.model.addConstr(in_flow == out_flow, name=f"flow_cons_{v}")

        for (i, j) in self.A_PD:
            self.model.addConstr(psi[i, j] <= self.inst.h[i, j], name=f"demand_bound_{i}_{j}")

        for (u_idx, v_idx) in self.A:
            available_k_components = []
            if (u_idx, v_idx, '0') in phi:
                available_k_components.append(phi[u_idx, v_idx, '0'])
            for k in self.E_uv.get((u_idx, v_idx), []):
                if (u_idx, v_idx, k) in phi:
                    available_k_components.append(phi[u_idx, v_idx, k])
            self.model.addConstr(psi[u_idx, v_idx] == gp.quicksum(available_k_components), name=f"flow_decomp_{u_idx}_{v_idx}")

        # 4. QoS and Linearized Virtual Wait
        for node in self.P + self.D:
            out_flow_u = gp.quicksum(psi[node, v_idx] for (u_idx, v_idx) in self.A if u_idx == node)
            self.model.addConstr(out_flow_u <= config.MAX_WAITING_FACTOR_P * x[node], name=f"qos_prob_{node}")

        for k in self.E:
            flow_through_k = gp.quicksum(phi[u_idx, v_idx, k] for (u_idx, v_idx) in self.A if (u_idx, v_idx, k) in phi)
            self.model.addConstr(flow_through_k <= config.MAX_FLOW_E * x[k], name=f"swap_activate_{k}")

        if config.USE_VIRTUAL_BASES_OPT:
            for lv in self.L_v:
                lhs = gp.quicksum((self.rho_vals[s] / (1 - self.rho_vals[s])) * u[lv, s] for s in self.S if self.rho_vals[s] < 1.0)
                rhs = config.MAX_WAITING_FACTOR_LV * gp.quicksum(psi[lv, i] for (l_idx, i) in self.A_LP if l_idx == lv)
                self.model.addConstr(lhs <= rhs, name=f"vir_wait_{lv}")

        # 5. Second-Order Cone Programming (SOCP) Reformulation for Queues
        for i_node in self.P + self.D:
            self.model.addConstr(lambd[i_node] == gp.quicksum(psi[i_node, v_idx] for (u_idx, v_idx) in self.A if u_idx == i_node), name=f"lambda_def_{i_node}")
            mu_i = self.inst.mu[i_node]
            self.model.addQConstr(4 * mu_i + (xi[i_node] + lambd[i_node] - mu_i + 1)**2 <= (xi[i_node] - lambd[i_node] + mu_i + 1)**2, name=f"socp_pd_queue_{i_node}")
            self.model.addConstr(xi[i_node] - lambd[i_node] + mu_i + 1 >= 0, name=f"socp_pd_queue_pos_{i_node}")

        # 6. Energy Balance (Node-based)
        # 6.1 Virtual Base Energy Balance (Exit from LV)
        if config.USE_VIRTUAL_BASES_OPT:
            for lv in self.L_v:
                wait_term = gp.quicksum((self.rho_vals[s] / (1 - self.rho_vals[s])) * u[lv, s] for s in self.S if self.rho_vals[s] < 1.0)
                transit_lhs = gp.LinExpr()
                replenish_rhs = gp.LinExpr()
                for (u_node, i_node) in self.A_LP:
                    if u_node == lv:
                        if (u_node, i_node, '0') in phi:
                            transit_lhs += phi[u_node, i_node, '0'] * self.inst.get_time(u_node, i_node)
                            # Direct flight does NOT replenish energy here based on LaTeX (sum over k in \tilde{E})
                        for k in self.E_uv.get((u_node, i_node), []):
                            if (u_node, i_node, k) in phi:
                                time_detour = self.inst.get_time(u_node, k) + self.inst.get_time(k, i_node)
                                transit_lhs += phi[u_node, i_node, k] * time_detour
                                replenish_rhs += phi[u_node, i_node, k]
                self.model.addConstr(config.ENERGY_RATE_OMEGA * (wait_term + transit_lhs) <= 
                                     (1 - config.ENERGY_SAFETY_MARGIN_EPSILON) * config.BATTERY_CAPACITY_OMEGA_UPPER * replenish_rhs, 
                                     name=f"energy_bal_lv_{lv}")

        # 6.2 Pickup Station Energy Balance (Exit from P)
        for i_node in self.P:
            transit_lhs = gp.LinExpr()
            replenish_rhs = gp.LinExpr()
            for (u_node, v_node) in self.A_PD:
                if u_node == i_node:
                    if (u_node, v_node, '0') in phi:
                        transit_lhs += phi[u_node, v_node, '0'] * self.inst.get_time(u_node, v_node)
                        # Direct flight does NOT replenish energy here
                    for k in self.E_uv.get((u_node, v_node), []):
                        if (u_node, v_node, k) in phi:
                            time_detour = self.inst.get_time(u_node, k) + self.inst.get_time(k, v_node)
                            transit_lhs += phi[u_node, v_node, k] * time_detour
                            replenish_rhs += phi[u_node, v_node, k]
            self.model.addConstr(config.ENERGY_RATE_OMEGA * (xi[i_node] + transit_lhs) <= 
                                 (1 - config.ENERGY_SAFETY_MARGIN_EPSILON) * config.BATTERY_CAPACITY_OMEGA_UPPER * replenish_rhs, 
                                 name=f"energy_bal_p_{i_node}")

        # # 6.3 Delivery Station Energy Balance (Exit from D)
        # for j_node in self.D:
        #     transit_lhs = gp.LinExpr()
        #     replenish_rhs = gp.LinExpr()
        #     for (u_node, v_node) in self.A:
        #         if u_node == j_node:
        #             if (u_node, v_node, '0') in phi:
        #                 transit_lhs += phi[u_node, v_node, '0'] * self.inst.get_time(u_node, v_node)
        #                 # Reaching a physical base counts as replenishment (phi^0_jl in LaTeX)
        #                 if v_node in self.L_hat:
        #                     replenish_rhs += phi[u_node, v_node, '0']
        #             for k in self.E_uv.get((u_node, v_node), []):
        #                 if (u_node, v_node, k) in phi:
        #                     time_detour = self.inst.get_time(u_node, k) + self.inst.get_time(k, v_node)
        #                     transit_lhs += phi[u_node, v_node, k] * time_detour
        #                     replenish_rhs += phi[u_node, v_node, k]
        #     self.model.addConstr(config.ENERGY_RATE_OMEGA * (xi[j_node] + transit_lhs) <= 
        #                          (1 - config.ENERGY_SAFETY_MARGIN_EPSILON) * config.BATTERY_CAPACITY_OMEGA_UPPER * replenish_rhs, 
        #                          name=f"energy_bal_d_{j_node}")

        # 7. Fleet Sizing and Budget
        def get_phi_time_expr():
            expr = gp.LinExpr()
            for (u_idx, v_idx) in self.A:
                if (u_idx, v_idx, '0') in phi:
                    expr += phi[u_idx, v_idx, '0'] * self.inst.get_time(u_idx, v_idx)
                for k in self.E_uv.get((u_idx, v_idx), []):
                    if (u_idx, v_idx, k) in phi:
                        time_detour = self.inst.get_time(u_idx, k) + self.inst.get_time(k, v_idx)
                        expr += phi[u_idx, v_idx, k] * time_detour
            return expr

        fleet_sum = gp.quicksum((self.rho_vals[s] / (1 - self.rho_vals[s])) * u[l, s] for l in self.L for s in self.S if self.rho_vals[s] < 1.0) + \
                    gp.quicksum(xi[i_node] for i_node in self.P + self.D) + \
                    get_phi_time_expr()
                    
        self.model.addConstr(fleet_sum <= gamma, name="fleet_calc")
        self.model.addConstr(gp.quicksum(self.inst.f[u_idx] * x[u_idx] for u_idx in self.N) + config.COST_DRONE * gamma <= config.BUDGET, name="budget")
        self.model.addConstr(gamma <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[l] for l in self.L_hat), name="gamma_limit")

    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal objective: {self.model.objVal:.4f}")
            return True
        return False

    def write_model(self, filename="model.lp"):
        self.model.write(filename)
