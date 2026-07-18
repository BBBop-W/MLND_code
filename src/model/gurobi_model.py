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

        self.A = sorted(self.inst.A)
        self.A_HH = sorted(self.inst.A_HH)
        self.A_LH = sorted(self.inst.A_LH)
        self.A_HS = sorted(self.inst.A_HS)
        self.A_SH = sorted(self.inst.A_SH)
        self.A_SL = sorted(self.inst.A_SL)
        self.A_SS = sorted(self.inst.A_SS)
        self.A_F = sorted(self.inst.A_F)
        self.A_virtual = sorted(set(self.A_HS) | set(self.A_SH))
        self.A_hat = sorted(set(self.A) - set(self.A_virtual))

        self.route_options = {}
        self.t = {}
        self.flight_cost = {}
        self.route_big_m = {}
        self._build_route_options()

        self.AY = []
        self.arc_cost = {}
        self._build_y_arcs()

        # Filled in build() for convenient downstream access.
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
                self.route_big_m[u, v, k] = self._route_flow_big_m(u, v, k)

    def _build_y_arcs(self):
        y_arcs = set(self.A_hat)
        for u, v in self.A_hat:
            for k in self.inst.E_uv.get((u, v), []):
                y_arcs.add((u, k))
                y_arcs.add((k, v))

        self.AY = sorted(y_arcs)
        for u, v in self.AY:
            self.arc_cost[u, v] = self._arc_activation_cost(u, v)

    def _route_flow_big_m(self, u, v, k):
        del u, v, k
        total_demand = sum(float(value) for value in self.inst.h.values())
        configured = float(getattr(config, "MAX_FLOW_E", 0.0))
        return max(1.0, configured, total_demand)

    def _arc_activation_cost(self, u, v):
        cost = getattr(config, "COST_ARC", 0.0)
        if isinstance(cost, dict):
            return float(cost.get((u, v), cost.get(f"{u},{v}", cost.get(f"{u}->{v}", 0.0))))
        return float(cost)

    def _node_activation(self, x, node):
        if node in self.S:
            return x[self.inst.s_to_h[node]]
        return x[node]

    def _hub_pool(self, h):
        return self.inst.h_to_s[h]

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
        tau_H = float(config.TAU_H)
        tau_S = float(config.TAU_S)
        omega = float(config.ENERGY_RATE_OMEGA)
        Omega = float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        energy_epsilon = float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))

        # Variables: x opens physical facilities, y opens non-virtual actual arcs,
        # z selects hub capacity levels, and Gamma is the integer fleet proxy.
        x = self.model.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        y = self.model.addVars(self.AY, vtype=GRB.BINARY, name="y")
        z = self.model.addVars(self.H, C, vtype=GRB.BINARY, name="z")
        Gamma = self.model.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")

        # Variables: continuous flow, utilization, queue-length, and auxiliary
        # service-rate variables.  H-S and S-H virtual arcs have psi only.
        rho_l = self.model.addVars(self.L, lb=0.0, name="rho_l")
        rho_s = self.model.addVars(self.S, lb=0.0, name="rho_s")
        alpha_keys = [
            (l, i, j)
            for (l, i) in self.A_LH
            for (i2, j) in self.A_HH
            if i2 == i
        ]
        alpha = self.model.addVars(alpha_keys, lb=0.0, name="alpha")

        beta_keys = list(self.A_HH)
        beta = self.model.addVars(beta_keys, lb=0.0, name="beta")

        psi = self.model.addVars(self.A, lb=0.0, name="psi")
        varphi_keys = [
            (u, v, k)
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        varphi = self.model.addVars(varphi_keys, lb=0.0, name="varphi")
        lambd = self.model.addVars(self.H, lb=0.0, name="lambda")
        mu = self.model.addVars(self.H, lb=0.0, name="mu")
        xi = self.model.addVars(self.HLS, lb=0.0, name="xi")

        # Objective: revenue minus station, capacity, actual-arc activation,
        # route operating, and fleet costs.  Virtual H-S/S-H arcs are not in AY.
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
        arc_activation_cost = gp.quicksum(
            self.arc_cost[u, v] * y[u, v] for (u, v) in self.AY
        )
        operating_cost = gp.quicksum(
            self.flight_cost[u, v, k] * varphi[u, v, k] for (u, v, k) in varphi_keys
        )

        self.model.setObjective(
            served_benefit
            - fixed_cost
            - arc_activation_cost
            - operating_cost
            - float(config.COST_DRONE) * Gamma,
            GRB.MAXIMIZE,
        )

        # (1) Base-to-hub assignment: sum_l y_li = x_i for every hub i.
        for i in self.H:
            assigned_bases = [l for (l, h) in self.A_LH if h == i]
            self.model.addConstr(
                gp.quicksum(y[l, i] for l in assigned_bases if (l, i) in y) == x[i],
                name=f"assign_{i}",
            )

        # (2) A base-to-hub assignment arc can open only if its base is open.
        for l, i in self.A_LH:
            self.model.addConstr(y[l, i] <= x[l], name=f"assign_active_{l}_{i}")

        # (3) Exactly one capacity level is selected when hub i is open.
        for i in self.H:
            self.model.addConstr(
                gp.quicksum(z[i, c] for c in C) == x[i],
                name=f"capacity_{i}",
            )

        # (4)-(5) Every non-LH actual arc in AY can open only when both endpoints
        # are active.  Holding-pool endpoints use the owner hub's x variable.
        for u, v in self.AY:
            if (u, v) in self.A_LH:
                continue
            self.model.addConstr(
                y[u, v] <= self._node_activation(x, u),
                name=f"arc_tail_active_{u}_{v}",
            )
            self.model.addConstr(
                y[u, v] <= self._node_activation(x, v),
                name=f"arc_head_active_{u}_{v}",
            )

        # (6) Base utilization can be positive only if the base is open.
        for l in self.L:
            self.model.addConstr(rho_l[l] <= x[l], name=f"rho_L_{l}")

        # (7) Holding-pool utilization can be positive only if its hub is open.
        for i, s in self.A_HS:
            self.model.addConstr(rho_s[s] <= x[i], name=f"rho_S_{s}")

        # (8)-(11) alpha_lij linearizes rho_l * y_li * y_ij.
        for l, i, j in alpha_keys:
            self.model.addConstr(alpha[l, i, j] <= rho_l[l], name=f"alpha_1_{l}_{i}_{j}")
            self.model.addConstr(alpha[l, i, j] <= y[l, i], name=f"alpha_2_{l}_{i}_{j}")
            self.model.addConstr(alpha[l, i, j] <= y[i, j], name=f"alpha_3_{l}_{i}_{j}")
            self.model.addConstr(
                alpha[l, i, j] >= rho_l[l] + y[l, i] + y[i, j] - 2,
                name=f"alpha_4_{l}_{i}_{j}",
            )

        # (12)-(14) beta_ij linearizes rho_s * y_ij for the pool s feeding i.
        for s, i in self.A_SH:
            for h, j in self.A_HH:
                if h != i:
                    continue
                self.model.addConstr(beta[i, j] <= rho_s[s], name=f"beta_1_{i}_{j}")
                self.model.addConstr(beta[i, j] <= y[i, j], name=f"beta_2_{i}_{j}")
                self.model.addConstr(
                    beta[i, j] >= rho_s[s] + y[i, j] - 1,
                    name=f"beta_3_{i}_{j}",
                )

        # (15) Loaded hub-to-hub flow generated by base and holding-pool chains.
        for i, j in self.A_HH:
            p_i = self._p_chain(i)
            demand = float(self.inst.h.get((i, j), 0.0))
            self.model.addConstr(
                psi[i, j]
                == demand
                * (
                    (1.0 - p_i)
                    * gp.quicksum(
                        alpha[l, i, j]
                        for (l, h) in self.A_LH
                        if h == i and (l, i, j) in alpha
                    )
                    + p_i * beta[i, j]
                ),
                name=f"psi_HH_{i}_{j}",
            )

        # (16) Empty base-to-hub repositioning flow feeding first-leg chains.
        for l, i in self.A_LH:
            self.model.addConstr(
                psi[l, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0))
                    * (1.0 - self._p_chain(i))
                    * alpha[l, i, j]
                    for (h, j) in self.A_HH
                    if h == i and (l, i, j) in alpha
                ),
                name=f"psi_LH_{l}_{i}",
            )

        # (17) Empty holding-pool-to-hub repositioning flow feeding chain legs.
        for s, i in self.A_SH:
            self.model.addConstr(
                psi[s, i]
                == gp.quicksum(
                    float(self.inst.h.get((i, j), 0.0))
                    * self._p_chain(i)
                    * beta[i, j]
                    for (h, j) in self.A_HH
                    if h == i
                ),
                name=f"psi_SH_{s}_{i}",
            )

        # (18) Flow conservation for bases, hubs, and holding pools.
        for q in self.HLS:
            self.model.addConstr(
                gp.quicksum(psi[u, q] for (u, v) in self.A if v == q)
                == gp.quicksum(psi[q, v] for (u, v) in self.A if u == q),
                name=f"flow_balance_{q}",
            )

        # (19) Route-option decomposition for every non-virtual aggregate arc.
        for u, v in self.A_hat:
            self.model.addConstr(
                gp.quicksum(
                    varphi[u, v, k]
                    for k in self.route_options.get((u, v), [])
                    if (u, v, k) in varphi
                )
                == psi[u, v],
                name=f"route_{u}_{v}",
            )

        # (20) A route flow can use aggregate arc (u,v) only if y_uv is open.
        # Direct option k=0 only needs this aggregate actual-arc activation.
        for u, v, k in varphi_keys:
            big_m = self.route_big_m[u, v, k]
            self.model.addConstr(
                varphi[u, v, k] <= big_m * y[u, v],
                name=f"route_arc_active_{u}_{v}_{k}",
            )

        # (21)-(22) A detour route through swapping station k can be used only
        # if both actual legs (u,k) and (k,v) are open.  These y-leg constraints
        # also imply the swapping station's x_k activation, so no extra phi-x
        # activation constraint is needed.
        for u, v, k in varphi_keys:
            if k not in self.E:
                continue
            big_m = self.route_big_m[u, v, k]
            self.model.addConstr(
                varphi[u, v, k] <= big_m * y[u, k],
                name=f"route_swap_in_active_{u}_{v}_{k}",
            )
            self.model.addConstr(
                varphi[u, v, k] <= big_m * y[k, v],
                name=f"route_swap_out_active_{u}_{v}_{k}",
            )

        total_queue_length = gp.quicksum(xi[u] for u in self.HLS)
        total_flight_time = gp.quicksum(
            self.t[u, v, k] * varphi[u, v, k] for (u, v, k) in varphi_keys
        )
        usable_battery = (1.0 - energy_epsilon) * Omega

        # (23) Base-side battery endurance: average energy of routes dispatched
        # from base l must be covered by routes that include a swap.
        for l in self.L:
            self.model.addConstr(
                omega
                * gp.quicksum(
                    self.t[l, i, k] * varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_hat
                )
                <= usable_battery
                * gp.quicksum(
                    varphi[l, i, k]
                    for (u, i, k) in varphi_keys
                    if u == l and (l, i) in self.A_hat and k in self.E
                ),
                name=f"energy_L_{l}",
            )

        # (24) Hub/holding-pool battery endurance for the chain leaving i and s.
        for i in self.H:
            s = self._hub_pool(i)
            self.model.addConstr(
                omega
                * (
                    xi[i]
                    + xi[s]
                    + gp.quicksum(
                        self.t[i, j, k] * varphi[i, j, k]
                        for (h, j, k) in varphi_keys
                        if h == i and (i, j) in self.A_HH
                    )
                    + gp.quicksum(
                        self.t[s, v, k] * varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                    )
                )
                <= usable_battery
                * (
                    gp.quicksum(
                        varphi[i, j, k]
                        for (h, j, k) in varphi_keys
                        if h == i and (i, j) in self.A_HH and k in self.E
                    )
                    + gp.quicksum(
                        varphi[s, v, k]
                        for (u, v, k) in varphi_keys
                        if u == s
                        and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                        and k in self.E
                    )
                ),
                name=f"energy_HS_{i}",
            )

        # (25) lambda_i equals total incoming aggregate flow to hub i.
        for i in self.H:
            self.model.addConstr(
                lambd[i]
                == gp.quicksum(psi[u, i] for (u, v) in self.A if v == i),
                name=f"lambda_{i}",
            )

        # Auxiliary definition for (26) and (29): mu_i = sum_c mu_c z_ic.
        for i in self.H:
            self.model.addConstr(
                mu[i] == gp.quicksum(float(mu_c[c]) * z[i, c] for c in C),
                name=f"mu_{i}",
            )

        # (26) Hub service stability lower bound.
        for i in self.H:
            self.model.addConstr(
                mu[i] - lambd[i] >= x[i] / tau_H,
                name=f"hub_wait_{i}",
            )

        # (27) Holding-pool mean dwell bound.
        for s in self.S:
            self.model.addConstr(
                xi[s] <= tau_S * gp.quicksum(psi[s, v] for (u, v) in self.A if u == s),
                name=f"mean_S_{s}",
            )

        # (28) SOC form of xi_u >= rho_u / (1 - rho_u) for bases and pools.
        q_nodes = self.L + self.S
        q_v1 = self.model.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v1")
        q_v2 = self.model.addVars(q_nodes, lb=-GRB.INFINITY, name="q_v2")
        q_t = self.model.addVars(q_nodes, lb=0.0, name="q_t")
        for q in q_nodes:
            rho_q = rho_l[q] if q in self.L else rho_s[q]
            self.model.addConstr(q_v1[q] == 2.0, name=f"q_v1_{q}")
            self.model.addConstr(q_v2[q] == xi[q] + rho_q, name=f"q_v2_{q}")
            self.model.addConstr(q_t[q] == xi[q] - rho_q + 2.0, name=f"q_t_{q}")
            q_name = f"queue_L_{q}" if q in self.L else f"queue_S_{q}"
            self.model.addQConstr(
                q_v1[q] * q_v1[q] + q_v2[q] * q_v2[q] <= q_t[q] * q_t[q],
                name=q_name,
            )

        # (29) SOC form of xi_i >= lambda_i / (mu_i - lambda_i) for hubs.
        h_v1 = self.model.addVars(self.H, lb=-GRB.INFINITY, name="h_v1")
        h_v2 = self.model.addVars(self.H, lb=-GRB.INFINITY, name="h_v2")
        h_t = self.model.addVars(self.H, lb=0.0, name="h_t")
        for i in self.H:
            sqrt_mu_z = gp.quicksum((float(mu_c[c]) ** 0.5) * z[i, c] for c in C)
            self.model.addConstr(h_v1[i] == 2.0 * sqrt_mu_z, name=f"h_v1_{i}")
            self.model.addConstr(h_v2[i] == xi[i] + lambd[i] - mu[i] + 1.0, name=f"h_v2_{i}")
            self.model.addConstr(h_t[i] == xi[i] - lambd[i] + mu[i] + 1.0, name=f"h_t_{i}")
            self.model.addQConstr(
                h_v1[i] * h_v1[i] + h_v2[i] * h_v2[i] <= h_t[i] * h_t[i],
                name=f"queue_H_{i}",
            )

        # (30) Fleet-size epigraph: queue length plus total physical flight time.
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
            "psi": psi,
            "varphi": varphi,
            "lambda": lambd,
            "mu": mu,
            "xi": xi,
            "Gamma": Gamma,
            "q_v1": q_v1,
            "q_v2": q_v2,
            "q_t": q_t,
            "h_v1": h_v1,
            "h_v2": h_v2,
            "h_t": h_t,
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
        f"OD={len(inst.h)}, A={len(inst.A)}, A_F={len(inst.A_F)}"
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
