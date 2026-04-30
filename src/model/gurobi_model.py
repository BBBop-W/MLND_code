import os
import sys

# Allow absolute imports when running this file directly.
if __name__ == "__main__":
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance


class MLNDModel:
    """Regime II: hub-side holding and continuous chaining."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.model = gp.Model("MLND_RegimeII_SOCP")
        self.model.setParam("Threads", 1)
        self._built = False

        self.L = list(self.inst.L)
        self.H = list(self.inst.H)
        self.S = list(self.inst.S)
        self.E = list(self.inst.E)
        self.N_x = self.L + self.H + self.E
        self.HLS = self.L + self.H + self.S

        self.A_HH = sorted(self.inst.A_HH)
        self.A_LH = sorted(self.inst.A_LH)
        self.A_HS = sorted(self.inst.A_HS)
        self.A_SH = sorted(self.inst.A_SH)
        self.A_SL = sorted(self.inst.A_SL)
        self.A_SS = sorted(self.inst.A_SS)
        self.A_phi = sorted(self.inst.A_phi)
        self.A_F = sorted(self.inst.A_F)

        self.route_options = {}
        self.t = {}
        self.flight_cost = {}
        self._build_route_options()

        # Filled in build() for convenient downstream access/debugging.
        self.vars = {}

    def _build_route_options(self):
        flight_cost_rate = float(getattr(config, "COST_FLIGHT_TIME", 0.1))

        for u, v in self.A_F:
            options = []
            if self.inst.get_dist(u, v) <= config.MAX_DIRECT_DISTANCE:
                options.append("0")
                self.t[u, v, "0"] = self.inst.get_time(u, v)

            for k in self.inst.E_uv.get((u, v), []):
                options.append(k)
                self.t[u, v, k] = self.inst.get_time(u, k) + self.inst.get_time(k, v)

            # Keep deterministic order and remove accidental duplicates.
            seen = set()
            ordered_options = []
            for k in options:
                if k not in seen:
                    ordered_options.append(k)
                    seen.add(k)

            self.route_options[u, v] = ordered_options
            for k in ordered_options:
                self.flight_cost[u, v, k] = flight_cost_rate * self.t[u, v, k]

    def _hub_pool(self, h):
        return self.inst.h_to_s[h]

    def _node_open_expr(self, node, x):
        if node in self.S:
            return x[self.inst.s_to_h[node]]
        return x[node]

    def _p_chain(self, hub):
        p_by_hub = getattr(config, "P_CHAIN_BY_HUB", None)
        if isinstance(p_by_hub, dict):
            return float(p_by_hub.get(hub, config.P_CHAIN))
        return float(config.P_CHAIN)

    def build(self):
        if self._built:
            return

        C = range(len(config.MU_HUB))
        mu_c = config.MU_HUB
        rho_max = float(config.RHO_MAX)
        tau_H = float(config.TAU_H)
        tau_S = float(config.TAU_S)
        omega = float(config.ENERGY_RATE_OMEGA)
        Omega = float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        swap_capacity = float(getattr(config, "SWAP_CAPACITY", config.MU_SWAPPING))
        max_flow = float(config.MAX_FLOW_E)

        x = self.model.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        z = self.model.addVars(self.H, C, vtype=GRB.BINARY, name="z")

        y_keys = [(i, l) for (l, i) in self.A_LH]
        y = self.model.addVars(y_keys, vtype=GRB.BINARY, name="y")

        rho_l = self.model.addVars(self.L, lb=0.0, ub=rho_max, name="rho_l")
        rho_s = self.model.addVars(self.S, lb=0.0, ub=rho_max, name="rho_s")

        alpha = self.model.addVars(y_keys, lb=0.0, ub=rho_max, name="alpha")
        beta_keys = [
            (l, i, j)
            for (l, i) in self.A_LH
            for (i2, j) in self.A_HH
            if i2 == i
        ]
        beta = self.model.addVars(beta_keys, lb=0.0, ub=rho_max, name="beta")

        delta_keys = list(self.A_HH)
        delta = self.model.addVars(delta_keys, lb=0.0, ub=rho_max, name="delta")

        eta = self.model.addVars(beta_keys, lb=0.0, name="eta")
        sigma = self.model.addVars(delta_keys, lb=0.0, name="sigma")
        psi = self.model.addVars(delta_keys, lb=0.0, name="psi")

        varphi = self.model.addVars(self.A_phi, lb=0.0, name="varphi")
        theta_keys = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        theta = self.model.addVars(theta_keys, lb=0.0, name="theta")

        lambd = self.model.addVars(self.H, lb=0.0, name="lambda")
        xi = self.model.addVars(self.HLS, lb=0.0, name="xi")
        Gamma = self.model.addVar(lb=0.0, name="Gamma")

        served_benefit = gp.quicksum(
            float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
            * psi[i, j]
            for i, j in self.A_HH
        )
        fixed_cost = gp.quicksum(
            float(self.inst.f.get(l, config.COST_BASE)) * x[l] for l in self.L
        )
        fixed_cost += gp.quicksum(
            float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i] for i in self.H
        )
        fixed_cost += gp.quicksum(
            float(self.inst.f.get(k, config.COST_SWAPPING)) * x[k] for k in self.E
        )
        fixed_cost += gp.quicksum(
            float(config.COST_HUB[c]) * z[i, c] for i in self.H for c in C
        )
        operating_cost = gp.quicksum(
            self.flight_cost[u, v, k] * theta[u, v, k] for (u, v, k) in theta_keys
        )

        self.model.setObjective(
            served_benefit
            - fixed_cost
            - operating_cost
            - float(config.COST_DRONE) * Gamma,
            GRB.MAXIMIZE,
        )

        # Assignment and capacity activation.
        for i in self.H:
            assigned_bases = [l for (l, h) in self.A_LH if h == i]
            self.model.addConstr(
                gp.quicksum(y[i, l] for l in assigned_bases if (i, l) in y) == x[i],
                name=f"assign_{i}",
            )
            self.model.addConstr(
                gp.quicksum(z[i, c] for c in C) == x[i],
                name=f"capacity_{i}",
            )

        for i, l in y_keys:
            self.model.addConstr(y[i, l] <= x[l], name=f"assign_active_{i}_{l}")

        for l in self.L:
            self.model.addConstr(rho_l[l] <= x[l], name=f"rho_L_{l}")
        for i in self.H:
            s = self._hub_pool(i)
            self.model.addConstr(rho_s[s] <= x[i], name=f"rho_S_{s}")

        # alpha[i,l] = rho_l[l] * y[i,l].
        for i, l in y_keys:
            self.model.addConstr(alpha[i, l] <= rho_l[l], name=f"alpha_1_{i}_{l}")
            self.model.addConstr(alpha[i, l] <= y[i, l], name=f"alpha_2_{i}_{l}")
            self.model.addConstr(
                alpha[i, l] >= rho_l[l] + y[i, l] - 1,
                name=f"alpha_3_{i}_{l}",
            )

        # beta[l,i,j] = alpha[i,l] * x_j.
        for l, i, j in beta_keys:
            self.model.addConstr(beta[l, i, j] <= alpha[i, l], name=f"beta_1_{l}_{i}_{j}")
            self.model.addConstr(beta[l, i, j] <= x[j], name=f"beta_2_{l}_{i}_{j}")
            self.model.addConstr(
                beta[l, i, j] >= alpha[i, l] + x[j] - 1,
                name=f"beta_3_{l}_{i}_{j}",
            )

        # delta[i,j] = rho_s(i) * x_j.
        for i, j in delta_keys:
            s = self._hub_pool(i)
            self.model.addConstr(delta[i, j] <= rho_s[s], name=f"delta_1_{i}_{j}")
            self.model.addConstr(delta[i, j] <= x[j], name=f"delta_2_{i}_{j}")
            self.model.addConstr(
                delta[i, j] >= rho_s[s] + x[j] - 1,
                name=f"delta_3_{i}_{j}",
            )

        for l, i, j in beta_keys:
            p_i = self._p_chain(i)
            demand = float(self.inst.h.get((i, j), 0.0))
            self.model.addConstr(
                eta[l, i, j] == (1.0 - p_i) * demand * beta[l, i, j],
                name=f"eta_{l}_{i}_{j}",
            )

        for i, j in delta_keys:
            p_i = self._p_chain(i)
            demand = float(self.inst.h.get((i, j), 0.0))
            self.model.addConstr(
                sigma[i, j] == p_i * demand * delta[i, j],
                name=f"sigma_{i}_{j}",
            )
            self.model.addConstr(
                psi[i, j]
                == gp.quicksum(eta[l, i, j] for (l, h) in self.A_LH if h == i and (l, i, j) in eta)
                + sigma[i, j],
                name=f"psi_{i}_{j}",
            )

        # Empty-drone source flows for base-side and holding-side initiations.
        for l, i in self.A_LH:
            self.model.addConstr(
                varphi[l, i]
                == gp.quicksum(eta[l, i, j] for (h, j) in self.A_HH if h == i and (l, i, j) in eta),
                name=f"varphi_LH_{l}_{i}",
            )

        for i in self.H:
            s = self._hub_pool(i)
            if (s, i) in varphi:
                self.model.addConstr(
                    varphi[s, i] == gp.quicksum(sigma[i, j] for (h, j) in self.A_HH if h == i),
                    name=f"varphi_SH_{s}_{i}",
                )

        for u, v in self.A_phi:
            self.model.addConstr(
                varphi[u, v] <= max_flow * self._node_open_expr(u, x),
                name=f"varphi_open_u_{u}_{v}",
            )
            self.model.addConstr(
                varphi[u, v] <= max_flow * self._node_open_expr(v, x),
                name=f"varphi_open_v_{u}_{v}",
            )

        for q in self.L + self.S:
            self.model.addConstr(
                gp.quicksum(varphi[u, q] for (u, v) in self.A_phi if v == q)
                == gp.quicksum(varphi[q, v] for (u, v) in self.A_phi if u == q),
                name=f"empty_balance_{q}",
            )

        for i in self.H:
            self.model.addConstr(
                gp.quicksum(varphi[u, i] for (u, v) in self.A_phi if v == i)
                == gp.quicksum(psi[i, j] for (h, j) in self.A_HH if h == i),
                name=f"H_empty_to_loaded_{i}",
            )
            self.model.addConstr(
                gp.quicksum(psi[j, i] for (j, h) in self.A_HH if h == i)
                == gp.quicksum(varphi[i, v] for (u, v) in self.A_phi if u == i),
                name=f"H_loaded_to_empty_{i}",
            )

        # Route-option decomposition for physical flights.
        for i, j in self.A_HH:
            self.model.addConstr(
                gp.quicksum(theta[i, j, k] for k in self.route_options.get((i, j), []) if (i, j, k) in theta)
                == psi[i, j],
                name=f"route_HH_{i}_{j}",
            )

        for u, v in self.A_LH + self.A_SL + self.A_SS:
            self.model.addConstr(
                gp.quicksum(theta[u, v, k] for k in self.route_options.get((u, v), []) if (u, v, k) in theta)
                == varphi[u, v],
                name=f"route_empty_{u}_{v}",
            )

        for k in self.E:
            self.model.addConstr(
                gp.quicksum(theta[u, v, k] for (u, v, kk) in theta_keys if kk == k)
                <= swap_capacity * x[k],
                name=f"swap_capacity_{k}",
            )

        total_queue_length = gp.quicksum(xi[u] for u in self.HLS)
        total_flight_time = gp.quicksum(
            self.t[u, v, k] * theta[u, v, k] for (u, v, k) in theta_keys
        )
        swapped_flights = gp.quicksum(
            theta[u, v, k] for (u, v, k) in theta_keys if k in self.E
        )
        self.model.addConstr(
            omega * (total_queue_length + total_flight_time) <= Omega * swapped_flights,
            name="energy_global",
        )

        for l in self.L:
            self.model.addConstr(
                omega
                * gp.quicksum(
                    self.t[l, i, k] * theta[l, i, k]
                    for (u, i, k) in theta_keys
                    if u == l and (l, i) in self.A_LH
                )
                <= Omega
                * gp.quicksum(
                    theta[l, i, k]
                    for (u, i, k) in theta_keys
                    if u == l and (l, i) in self.A_LH and k in self.E
                ),
                name=f"energy_L_{l}",
            )

        for i in self.H:
            self.model.addConstr(
                lambd[i]
                == gp.quicksum(psi[i, j] for (h, j) in self.A_HH if h == i)
                + gp.quicksum(psi[j, i] for (j, h) in self.A_HH if h == i),
                name=f"lambda_{i}",
            )
            mu_z = gp.quicksum(float(mu_c[c]) * z[i, c] for c in C)
            self.model.addConstr(lambd[i] <= rho_max * mu_z, name=f"hub_stability_{i}")

        for l in self.L:
            self.model.addQConstr(
                4 + (xi[l] + rho_l[l]) * (xi[l] + rho_l[l])
                <= (xi[l] - rho_l[l] + 2) * (xi[l] - rho_l[l] + 2),
                name=f"queue_L_{l}",
            )

        for s in self.S:
            self.model.addQConstr(
                4 + (xi[s] + rho_s[s]) * (xi[s] + rho_s[s])
                <= (xi[s] - rho_s[s] + 2) * (xi[s] - rho_s[s] + 2),
                name=f"queue_S_{s}",
            )

        for i in self.H:
            mu_z = gp.quicksum(float(mu_c[c]) * z[i, c] for c in C)
            self.model.addQConstr(
                4 * mu_z + (xi[i] + lambd[i] - mu_z + 1) * (xi[i] + lambd[i] - mu_z + 1)
                <= (xi[i] - lambd[i] + mu_z + 1) * (xi[i] - lambd[i] + mu_z + 1),
                name=f"queue_H_{i}",
            )
            self.model.addConstr(xi[i] <= tau_H * lambd[i], name=f"mean_H_{i}")

        for s in self.S:
            self.model.addConstr(
                xi[s] <= tau_S * gp.quicksum(varphi[s, v] for (u, v) in self.A_phi if u == s),
                name=f"mean_S_{s}",
            )

        self.model.addConstr(
            total_queue_length + total_flight_time <= Gamma,
            name="fleet",
        )

        self.vars = {
            "x": x,
            "z": z,
            "y": y,
            "rho_l": rho_l,
            "rho_s": rho_s,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "eta": eta,
            "sigma": sigma,
            "psi": psi,
            "varphi": varphi,
            "theta": theta,
            "lambda": lambd,
            "xi": xi,
            "Gamma": Gamma,
        }
        self._built = True

    def solve(self):
        self.build()
        self.apply_solver_config()
        self.model.optimize()
        if self.model.SolCount > 0:
            if self.model.status == GRB.OPTIMAL:
                print(f"Optimal objective: {self.model.objVal:.4f}")
            else:
                print(
                    f"Best objective: {self.model.objVal:.4f} "
                    f"(status={self.model.status}, solutions={self.model.SolCount})"
                )
            return True
        return False

    def apply_solver_config(self):
        time_limit = getattr(config, "MODEL_TIME_LIMIT", None)
        mip_gap = getattr(config, "MODEL_MIP_GAP", None)
        if time_limit is not None:
            self.model.setParam("TimeLimit", float(time_limit))
        if mip_gap is not None:
            self.model.setParam("MIPGap", float(mip_gap))

    def write_model(self, filename="model.lp"):
        self.model.write(filename)


def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "n"}


def run_meituan_regime_ii(
    generate_sample=None,
    date_filter=None,
    project_root=None,
    save_outputs=True,
    plot_network=True,
):
    """Generate/load the Meituan instance, solve Regime II, and save outputs."""
    from src.data_processor.generator import DataGenerator
    from src.data_processor.loader import DataLoader
    from src.utils.output_processor import ResultProcessor

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if generate_sample is None:
        generate_sample = _env_flag("MLND_GENERATE_SAMPLE", True)
    if date_filter is None:
        date_filter = config.MEITUAN_DEMAND_DATES[0]

    period_id = f"h{config.MEITUAN_DEMAND_HOUR:02d}"
    meituan_instance_dir = os.path.join(project_root, "data", "instances", "Meituan")
    meituan_results_dir = os.path.join(project_root, "results", "Meituan")
    meituan_raw_file = os.path.join(project_root, "Meituan-TSL", "all_waybill_info_meituan_0322.csv")
    if save_outputs:
        os.makedirs(meituan_results_dir, exist_ok=True)

    print(
        f"Regime II run: generate={generate_sample}, period={period_id}, "
        f"time_limit={config.MODEL_TIME_LIMIT}, mip_gap={config.MODEL_MIP_GAP}"
    )

    if generate_sample:
        DataGenerator(seed=42).generate_meituan_hourly(
            raw_data_path=meituan_raw_file,
            save_dir=meituan_instance_dir,
            num_hubs=config.MEITUAN_NUM_HUBS,
            num_bases=config.MEITUAN_NUM_BASES,
            num_swapping=config.MEITUAN_NUM_SWAPPING,
            date_filter=date_filter,
            map_half_size_km=config.MEITUAN_MAP_HALF_SIZE_KM,
            candidate_date_filter=config.MEITUAN_CANDIDATE_DATES,
            demand_date_filter=config.MEITUAN_DEMAND_DATES,
            candidate_hour_start=config.MEITUAN_CANDIDATE_HOUR_START,
            candidate_hour_end=config.MEITUAN_CANDIDATE_HOUR_END,
            demand_hour_start=config.MEITUAN_DEMAND_HOUR,
            colocate_base_swaps=config.MEITUAN_COLOCATE_BASE_SWAPS,
        )

    period_path = os.path.join(meituan_instance_dir, "orders", f"{period_id}.csv")
    load_period_id = period_id if os.path.exists(period_path) else None
    inst = DataLoader().load_instance_dir(meituan_instance_dir, period_id=load_period_id)
    print(
        f"Loaded: H={len(inst.H)}, L={len(inst.L)}, S={len(inst.S)}, E={len(inst.E)}, "
        f"OD={len(inst.h)}, A_F={len(inst.A_F)}, A_phi={len(inst.A_phi)}"
    )

    model = MLNDModel(inst)
    success = model.solve()

    if success or model.model.SolCount > 0:
        processor = ResultProcessor(model, inst)
        processor.print_summary()
        if save_outputs:
            model.write_model(os.path.join(meituan_results_dir, "model_regime_ii.lp"))
            processor.save_node_flow_summary(os.path.join(meituan_results_dir, "node_flow_summary.csv"))
            processor.save_open_facilities_config(os.path.join(meituan_results_dir, "open_facilities.csv"))
            processor.save_python_object(os.path.join(meituan_results_dir, "result_data.pkl"))
            if plot_network:
                processor.plot_network(save_path=os.path.join(meituan_results_dir, "network_plot.png"))
            print(f"Saved outputs to {meituan_results_dir}")
    else:
        print("No feasible solution found.")

    return model


if __name__ == "__main__":
    run_meituan_regime_ii()
