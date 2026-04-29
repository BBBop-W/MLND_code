import sys
import os
import math

# Allow absolute imports when running as main
if __name__ == "__main__":
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

import gurobipy as gp
from gurobipy import GRB
from src.data_processor.instance import Instance
from src import config

class MLNDModel:
    def __init__(self, instance: Instance):
        self.inst = instance
        self.model = gp.Model("MLND_SOCP")
        self.model.setParam('Threads', 1)
        
        self.L = self.inst.L # physical bases
        self.H = self.inst.H # Hubs
        self.E = self.inst.E # Swapping stations
        
        self.L_v = self.inst.L_v
        self.L_tilde = self.L + self.L_v
            
        # x decisions are only for physical nodes (H, L, E).
        self.N_x = self.H + self.L + self.E
        # Arc traversals may still involve virtual nodes.
        self.N_all = self.inst.N

        # Arcs
        self.A = [(u, v) for (u, v) in self.inst.A]
        self.A_HH = [(u, v) for (u, v) in self.inst.A_HH if (u, v) in self.A]
        self.A_LH = [(u, v) for (u, v) in self.inst.A_LH if (u, v) in self.A]
        self.A_HL = [(u, v) for (u, v) in self.inst.A_HL if (u, v) in self.A]
        self.A_M = [(i, self.inst.h_to_lv[i]) for i in self.H if i in self.inst.h_to_lv] # M mapping
        
        self.E_uv = {arc: self.inst.E_uv[arc] for arc in self.A if arc in self.inst.E_uv}
        # pre-compute t_uv^k
        self.t = {}
        for (u, v) in self.A:
            if self.inst.get_dist(u, v) <= config.MAX_DIRECT_DISTANCE:
                self.t[u, v, '0'] = self.inst.get_time(u, v)
            for k in self.E_uv.get((u, v), []):
                self.t[u, v, k] = self.inst.get_time(u, k) + self.inst.get_time(k, v)

    def build(self):
        C = range(len(config.MU_HUB)) 
        mu_c = config.MU_HUB
        f_hub = config.COST_HUB
        p = config.P_CHAIN
        rho_max = config.RHO_MAX
        tau_H = config.TAU_H
        tau_v = config.TAU_LV
        gamma_param = config.GAMMA
        omega = config.ENERGY_RATE_OMEGA
        Omega_upper = config.BATTERY_CAPACITY_OMEGA_UPPER
        epsilon = config.ENERGY_SAFETY_MARGIN_EPSILON
        
        x = self.model.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        z = self.model.addVars(self.H, C, vtype=GRB.BINARY, name="z")
        
        y_keys = [(i, l) for i in self.H for l in self.L]
        y = self.model.addVars(y_keys, vtype=GRB.BINARY, name="y")
        
        rho_l = self.model.addVars(self.L, lb=0.0, ub=1.0, name="rho_l")
        rho_ell = self.model.addVars(self.L_v, lb=0.0, ub=1.0, name="rho_ell")
        rho_tilde = self.model.addVars(y_keys, lb=0.0, ub=1.0, name="rho_tilde")

        w_keys = [(i, j) for (i, j) in self.A_HH]
        w = self.model.addVars(w_keys, lb=0.0, name="w")
        w_base_keys = [(i, l, j) for (i, j) in self.A_HH for l in self.L]
        w_base = self.model.addVars(w_base_keys, lb=0.0, name="w_base")
        w_virt = self.model.addVars(w_keys, lb=0.0, name="w_virt")
        psi = self.model.addVars(w_keys, lb=0.0, name="psi")
        
        phi_keys = []
        for l in self.L_tilde:
            for i in self.H:
                if (l, i) in self.A or (i, l) in self.A:
                    phi_keys.append((l, i))
                    phi_keys.append((i, l))
            for ell in self.L_v:
                if (l, ell) in self.A or (ell, l) in self.A:
                    phi_keys.append((l, ell))
                    phi_keys.append((ell, l))

        phi_keys = list(set(phi_keys)) # Remove duplicates
        phi = self.model.addVars(phi_keys, lb=0.0, name="phi")

        # Node-open indicator used to gate phi arcs, including virtual nodes.
        node_open = {}
        for l in self.L:
            node_open[l] = x[l]
        for h in self.H:
            node_open[h] = x[h]
        for ell in self.L_v:
            owner_h = self.inst.lv_to_h.get(ell)
            if owner_h is not None:
                node_open[ell] = x[owner_h]
        
        theta_keys = [(u, v, k) for (u, v, k) in self.t.keys()]
        theta = self.model.addVars(theta_keys, lb=0.0, name="theta")
        
        lambd = self.model.addVars(self.H, lb=0.0, name="lambda")
        xi = self.model.addVars(self.H + self.L_tilde, lb=0.0, name="xi")
        Gamma_var = self.model.addVar(lb=0.0, name="Gamma")

        r = config.ORDER_REVENUE_COEF
        served_demand = gp.quicksum(psi[i, j] for (i, j) in self.A_HH)
        fixed_cost = gp.quicksum(self.inst.f.get(u, config.COST_BASE) * x[u] for u in self.L + self.E)
        fixed_cost += gp.quicksum(f_hub[c] * z[i, c] for i in self.H for c in C)
        flight_coef = getattr(config, "COST_FLIGHT_TIME", 0.1)
        flight_cost = gp.quicksum(theta[u, v, k] * flight_coef for (u, v, k) in theta_keys)
        
        self.model.setObjective(
            r * served_demand - flight_cost - fixed_cost - config.COST_DRONE * Gamma_var,
            GRB.MAXIMIZE
        )

        # 1-3
        for i in self.H:
            self.model.addConstr(gp.quicksum(y[i, l] for l in self.L) == x[i], name=f"single_assign_{i}")
            self.model.addConstr(gp.quicksum(z[i, c] for c in C) == x[i], name=f"capacity_select_{i}")
            for l in self.L:
                self.model.addConstr(y[i, l] <= x[l], name=f"assign_x_{i}_{l}")
                
        # 4-5
        for l in self.L:
            self.model.addConstr(rho_l[l] <= x[l], name=f"rho_base_bound_{l}")
        for i, ell in self.A_M:
            self.model.addConstr(rho_ell[ell] <= x[i], name=f"rho_virt_bound_{i}")

        # 6-7
        for i in self.H:
            for l in self.L:
                self.model.addConstr(rho_tilde[i, l] <= rho_l[l], name=f"lin_rho_1a_{i}_{l}")
                self.model.addConstr(rho_tilde[i, l] <= y[i, l], name=f"lin_rho_1b_{i}_{l}")
                self.model.addConstr(rho_tilde[i, l] >= rho_l[l] + y[i, l] - 1, name=f"lin_rho_2_{i}_{l}")

        # 8-10
        # Split linearization by source:
        #   w_base[i,l,j] = rho_tilde[i,l] * x[j]
        #   w_virt[i,j]   = rho_ell[ell(i)] * x[j]
        #   w[i,j]        = (1-p) * sum_l w_base[i,l,j] + p * w_virt[i,j]
        for i, j in self.A_HH:
            ell = self.inst.h_to_lv.get(i)
            for l in self.L:
                self.model.addConstr(w_base[i, l, j] <= rho_tilde[i, l], name=f"lin_wb_1a_{i}_{l}_{j}")
                self.model.addConstr(w_base[i, l, j] <= x[j], name=f"lin_wb_1b_{i}_{l}_{j}")
                self.model.addConstr(w_base[i, l, j] >= rho_tilde[i, l] + x[j] - 1, name=f"lin_wb_2_{i}_{l}_{j}")

            if ell:
                self.model.addConstr(w_virt[i, j] <= rho_ell[ell], name=f"lin_wv_1a_{i}_{j}")
                self.model.addConstr(w_virt[i, j] <= x[j], name=f"lin_wv_1b_{i}_{j}")
                self.model.addConstr(w_virt[i, j] >= rho_ell[ell] + x[j] - 1, name=f"lin_wv_2_{i}_{j}")
                self.model.addConstr(
                    w[i, j] == (1 - p) * gp.quicksum(w_base[i, l, j] for l in self.L) + p * w_virt[i, j],
                    name=f"lin_w_mix_{i}_{j}",
                )
            else:
                self.model.addConstr(w_virt[i, j] == 0.0, name=f"lin_wv_zero_{i}_{j}")
                self.model.addConstr(
                    w[i, j] == (1 - p) * gp.quicksum(w_base[i, l, j] for l in self.L),
                    name=f"lin_w_mix_{i}_{j}",
                )

            self.model.addConstr(psi[i, j] == self.inst.h.get((i, j), 0) * w[i, j], name=f"lin_psi_{i}_{j}")
            
            
            
        # 11-13
        for i in self.H:
            self.model.addConstr(
                gp.quicksum(phi[l, i] for l in self.L_tilde if (l, i) in phi_keys) == 
                gp.quicksum(psi[i, j] for j in self.H if (i, j) in self.A_HH), 
                name=f"hub_dept_{i}")
            
            ell = self.inst.h_to_lv.get(i)
            for l in self.L:
                if (l, i) in phi_keys:
                    self.model.addConstr(
                        phi[l, i] == (1 - p) * gp.quicksum(
                            self.inst.h.get((i, j), 0.0) * w_base[i, l, j]
                            for j in self.H
                            if (i, j) in self.A_HH
                        ),
                        name=f"phi_base_{l}_{i}",
                    )

            if ell and (ell, i) in phi_keys:
                self.model.addConstr(
                    phi[ell, i] == p * gp.quicksum(
                        self.inst.h.get((i, j), 0.0) * w_virt[i, j]
                        for j in self.H
                        if (i, j) in self.A_HH
                    ),
                    name=f"phi_virt_{i}",
                )

                
        for i, ell in self.A_M:
            self.model.addConstr(
                gp.quicksum(psi[j, i] for j in self.H if (j, i) in self.A_HH) == phi[i, ell], 
                name=f"hub_arrival_{i}")
            self.model.addConstr(
                gp.quicksum(phi[u_node, ell] for u_node in self.H + self.L_v if (u_node, ell) in phi_keys)
                == gp.quicksum(phi[ell, v_node] for v_node in self.H + self.L_tilde if (ell, v_node) in phi_keys),
                name=f"virt_bal_{ell}")

        # Gate all phi arcs by endpoint activation so closed hubs/virtual nodes
        # cannot carry circulation flow.
        for (u_node, v_node) in phi_keys:
            if u_node in node_open:
                self.model.addConstr(
                    phi[u_node, v_node] <= config.MAX_FLOW_E * node_open[u_node],
                    name=f"phi_open_u_{u_node}_{v_node}",
                )
            if v_node in node_open:
                self.model.addConstr(
                    phi[u_node, v_node] <= config.MAX_FLOW_E * node_open[v_node],
                    name=f"phi_open_v_{u_node}_{v_node}",
                )
            
        for l in self.L:
            self.model.addConstr(
                gp.quicksum(phi[l, i] for i in self.H + self.L_tilde if (l, i) in phi_keys) <= x[l] * config.MAX_FLOW_E,
                name=f"base_out_{l}")
            self.model.addConstr(
                gp.quicksum(phi[i, l] for i in self.H + self.L_tilde if (i, l) in phi_keys) <= x[l] * config.MAX_FLOW_E,
                name=f"base_in_{l}")

        # 14-16 decomp
        for i, j in self.A_HH:
            ell = self.inst.h_to_lv.get(i)
            v_v = self.inst.h_to_lv.get(j)
            if ell and v_v:
                k_sum = gp.quicksum(theta[i, j, k] for k in ["0"] + self.inst.E_uv.get((i,j), []) if (i, j, k) in theta)
                self.model.addConstr(k_sum == psi[i, j] + (phi[ell, v_v] if (ell, v_v) in phi_keys else 0), name=f"decomp_hub_hub_{i}_{j}")
                
        for i, ell in self.A_M:
            if (i, ell) in self.A:
                k_sum_iell = gp.quicksum(
                    theta[i, ell, k]
                    for k in ["0"] + self.inst.E_uv.get((i, ell), [])
                    if (i, ell, k) in theta
                )
                self.model.addConstr(
                    k_sum_iell == (phi[i, ell] if (i, ell) in phi_keys else 0),
                    name=f"decomp_hub_virt_{i}_{ell}",
                )
            if (ell, i) in self.A:
                k_sum_elli = gp.quicksum(
                    theta[ell, i, k]
                    for k in ["0"] + self.inst.E_uv.get((ell, i), [])
                    if (ell, i, k) in theta
                )
                self.model.addConstr(
                    k_sum_elli == (phi[ell, i] if (ell, i) in phi_keys else 0),
                    name=f"decomp_virt_hub_{ell}_{i}",
                )

            for l in self.L:
                if (l, i) in self.A:
                    k_sum_li = gp.quicksum(theta[l, i, k] for k in ["0"] + self.inst.E_uv.get((l,i), []) if (l, i, k) in theta)
                    self.model.addConstr(k_sum_li == (phi[l, i] if (l, i) in phi_keys else 0), name=f"decomp_base_hub_{l}_{i}")
                if (ell, l) in self.A:
                    k_sum_il = gp.quicksum(theta[ell, l, k] for k in ["0"] + self.inst.E_uv.get((ell,l), []) if (ell, l, k) in theta)
                    self.model.addConstr(k_sum_il == (phi[ell, l] if (ell, l) in phi_keys else 0), name=f"decomp_hub_base_{i}_{l}")

        # Theta-level virtual-node conservation for consistent flown-leg accounting.
        for ell in self.L_v:
            in_theta = gp.quicksum(
                theta[u_node, ell, k]
                for (u_node, v_node) in self.A
                if v_node == ell
                for k in ["0"] + self.inst.E_uv.get((u_node, v_node), [])
                if (u_node, v_node, k) in theta
            )
            out_theta = gp.quicksum(
                theta[ell, v_node, k]
                for (u_node, v_node) in self.A
                if u_node == ell
                for k in ["0"] + self.inst.E_uv.get((u_node, v_node), [])
                if (u_node, v_node, k) in theta
            )
            self.model.addConstr(in_theta == out_theta, name=f"theta_bal_{ell}")

        # 17-20
        for i, ell in self.A_M:
            sum_psi = gp.quicksum(psi[j, i] for j in self.H if (j, i) in self.A_HH)
            sum_phi_l = gp.quicksum(phi[l, i] for l in self.L if (l, i) in phi_keys)
            phi_elli = phi[ell, i] if (ell, i) in phi_keys else 0
            self.model.addConstr(lambd[i] == sum_psi + sum_phi_l + phi_elli, name=f"hub_thr_{i}")
            
            self.model.addConstr(lambd[i] <= rho_max * gp.quicksum(mu_c[c] * z[i, c] for c in C), name=f"qos_hub_stab_{i}")
            
            if gamma_param not in (0, 1):
                self.model.addConstr(xi[i] <= (-tau_H / math.log(gamma_param)) * lambd[i], name=f"qos_hub_exp_{i}")
                
                sum_phi_iell = phi[i, ell] if (i, ell) in phi_keys else 0
                sum_phi_vell = gp.quicksum(phi[v, ell] for v in self.L_v if (v, ell) in phi_keys)
                self.model.addConstr(xi[ell] <= (-tau_v / math.log(gamma_param)) * (sum_phi_iell + sum_phi_vell), name=f"qos_virt_{ell}")

        # 21-22 Energy
        for i, ell in self.A_M:
            out_arcs_i = [(i_u, i_v) for (i_u, i_v) in self.A if i_u == i or i_u == ell]
            lhs = omega * (
                xi[i]
                + xi[ell]
                + gp.quicksum(
                    theta[i_u, v, k] * self.t[i_u, v, k]
                    for (i_u, v) in out_arcs_i
                    for k in ["0"] + self.inst.E_uv.get((i_u, v), [])
                    if (i_u, v, k) in theta and (i_u, v) not in self.A_M
                )
            )
            rhs = (1 - epsilon) * Omega_upper * gp.quicksum(
                theta[i_u, v, k]
                for (i_u, v) in out_arcs_i
                for k in self.inst.E_uv.get((i_u, v), [])
                if (i_u, v, k) in theta and (i_u, v) not in self.A_M
            )
            self.model.addConstr(lhs <= rhs, name=f"energy_hub_{i}")
            
        for l in self.L:
            lhs = omega * gp.quicksum(theta[l, i, k] * self.t[l, i, k] for i in self.H if (l, i) in self.A for k in ["0"] + self.inst.E_uv.get((l,i), []) if (l, i, k) in theta)
            rhs = (1 - epsilon) * Omega_upper * gp.quicksum(theta[l, i, k] for i in self.H if (l, i) in self.A for k in self.inst.E_uv.get((l,i), []) if (l, i, k) in theta)
            self.model.addConstr(lhs <= rhs, name=f"energy_base_{l}")

        # 23 Swap capacity
        for k in self.E:
            sum_th = gp.quicksum(theta[u, v, k] for (u, v) in self.A if (u, v, k) in theta)
            self.model.addConstr(sum_th <= config.MAX_FLOW_E * x[k], name=f"swap_cap_{k}")

        # 24 Fleet
        fleet_expr = gp.quicksum(xi[u] for u in self.L_tilde + self.H) + gp.quicksum(theta[u, v, k] * self.t[u, v, k] for (u, v, k) in theta_keys)
        self.model.addConstr(fleet_expr <= Gamma_var, name="fleet_size")

        # 25-26 SOCP Wait Approximations
        for u in self.L_tilde:
            ru = rho_ell[u] if u in self.L_v else rho_l[u]
            self.model.addQConstr(4  + (xi[u] + ru)**2 <= (xi[u] - ru + 2)**2, name=f"socp_base_{u}")
            
        for i in self.H:
            mu_z = gp.quicksum(mu_c[c] * z[i, c] for c in C)
            self.model.addQConstr(4 * mu_z + (xi[i] + lambd[i] - mu_z + 1)**2 <= (xi[i] - lambd[i] + mu_z + 1)**2, name=f"socp_hub_{i}")

    def solve(self):
        self.build() 
        self.model.optimize()
        self.write_model("debug_gurobi_model.lp")
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal objective: {self.model.objVal:.4f}")
            return True
        return False

    def write_model(self, filename="model.lp"):
        self.model.write(filename)

if __name__ == "__main__":
    from src.data_processor.generator import DataGenerator
    from src.data_processor.loader import DataLoader

    meituan_instance_dir = os.path.join(_proj_root, "data", "instances", "Meituan")
    meituan_results_dir = os.path.join(_proj_root, "results", "Meituan")
    os.makedirs(meituan_results_dir, exist_ok=True)

    meituan_raw_file = os.path.join(_proj_root, "Meituan-TSL",
                                    "all_waybill_info_meituan_0322.csv")
    date_filter=20221017
    
    print("=== 1. Generating Test Data ===")
    gen = DataGenerator(seed=42)
    gen.generate_meituan_hourly(
            raw_data_path=meituan_raw_file,
            save_dir=meituan_instance_dir,
            num_hubs=config.MEITUAN_NUM_HUBS,
            num_bases=config.MEITUAN_NUM_BASES,
            num_swapping=config.MEITUAN_NUM_SWAPPING,
            date_filter=date_filter,
            map_half_size_km=config.MEITUAN_MAP_HALF_SIZE_KM,
            min_node_hourly_orders=config.MEITUAN_MIN_NODE_HOURLY_ORDERS,
        )
    
    print("\n=== 2. Loading Instance ===")
    loader = DataLoader()
    inst = loader.load_instance(
        os.path.join(meituan_instance_dir, "locations.csv"),
        os.path.join(meituan_instance_dir, "orders.csv")
    )
    
    print(f"Loaded: |H|={len(inst.H)}, |L|={len(inst.L)}, |E|={len(inst.E)}, |A_HH|={len(inst.A_HH)}")
    
    print("\n=== 3. Building & Solving Model ===")
    m = MLNDModel(inst)
    m.solve()
    m.write_model(os.path.join(_proj_root, "debug_gurobi_model.lp"))
