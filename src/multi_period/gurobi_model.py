import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

# Allow absolute imports when running this file directly.
if __package__ in (None, ""):
    _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

import gurobipy as gp
from gurobipy import GRB

from src import config
from src.data_processor.instance import Instance
from src.data_processor.loader import DataLoader


Demand = Dict[Tuple[str, str], float]


@dataclass
class MultiPeriodInstance:
    """Shared physical network with period-specific hub-to-hub demand."""

    base: Instance
    periods: List[str]
    demand: Dict[str, Demand]
    weights: Dict[str, float]


def _period_id_from_hour(hour: int) -> str:
    return f"h{int(hour):02d}"


def _as_period_ids(period_ids=None, demand_hours=None) -> List[str]:
    if period_ids is not None:
        values = period_ids
    else:
        if demand_hours is None:
            demand_hours = getattr(config, "MEITUAN_DEMAND_HOURS", [12, 17, 20])
        values = [_period_id_from_hour(h) if isinstance(h, int) else str(h) for h in demand_hours]

    if isinstance(values, (str, int)):
        values = [values]

    period_list = []
    for value in values:
        if isinstance(value, int):
            period_list.append(_period_id_from_hour(value))
        else:
            period_list.append(str(value))

    if not period_list:
        raise ValueError("At least one period is required.")
    return period_list


def _weights_by_period(periods: Sequence[str], period_weights=None) -> Dict[str, float]:
    if period_weights is None:
        return {t: 1.0 for t in periods}

    if isinstance(period_weights, dict):
        weights = {str(t): float(w) for t, w in period_weights.items()}
        return {t: weights.get(t, 1.0) for t in periods}

    weights_list = [float(w) for w in period_weights]
    if len(weights_list) != len(periods):
        raise ValueError("period_weights must have the same length as periods.")
    return dict(zip(periods, weights_list))


def _validate_same_locations(reference: Instance, other: Instance, period_id: str) -> None:
    for attr in ("H", "L", "E"):
        if list(getattr(reference, attr)) != list(getattr(other, attr)):
            raise ValueError(f"Period {period_id} has different {attr} nodes.")

    for node, coord in reference.coords.items():
        if node.startswith("S_"):
            continue
        other_coord = other.coords.get(node)
        if other_coord is None:
            raise ValueError(f"Period {period_id} is missing node {node}.")
        if max(abs(coord[0] - other_coord[0]), abs(coord[1] - other_coord[1])) > 1e-7:
            raise ValueError(f"Period {period_id} has different coordinates for node {node}.")


def load_multi_period_instance(
    instance_dir: str,
    periods: Sequence[str],
    period_weights=None,
) -> MultiPeriodInstance:
    """Load shared locations and one orders file per period."""
    loader = DataLoader()
    period_list = [str(t) for t in periods]
    loaded: Dict[str, Instance] = {}

    for period_id in period_list:
        orders_path = os.path.join(instance_dir, "orders", f"{period_id}.csv")
        if not os.path.exists(orders_path):
            raise FileNotFoundError(f"Missing period orders file: {orders_path}")
        loaded[period_id] = loader.load_instance_dir(instance_dir, period_id=period_id)

    base = loaded[period_list[0]]
    for period_id in period_list[1:]:
        _validate_same_locations(base, loaded[period_id], period_id)

    demand = {period_id: dict(inst.h) for period_id, inst in loaded.items()}
    union_demand: Demand = {}
    for demand_t in demand.values():
        for key, value in demand_t.items():
            union_demand[key] = max(union_demand.get(key, 0.0), float(value))

    base.h = union_demand
    base.A_HH = set(union_demand)
    base.parse_for_gurobi()

    return MultiPeriodInstance(
        base=base,
        periods=period_list,
        demand=demand,
        weights=_weights_by_period(period_list, period_weights),
    )


class MultiPeriodMLNDModel:
    """Regime II MISOCP with shared facility decisions and period-specific flows."""

    def __init__(self, mp_instance: MultiPeriodInstance):
        self.mp = mp_instance
        self.inst = mp_instance.base
        self.model = gp.Model("MLND_RegimeII_MultiPeriod_SOCP")
        self.model.setParam("Threads", 1)
        self._built = False

        self.T = list(mp_instance.periods)
        self.period_weights = dict(mp_instance.weights)
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

        self.route_options = {}
        self.t = {}
        self.flight_cost = {}
        self._build_route_options()

        self.vars = {}

    def _build_route_options(self) -> None:
        flight_cost_rate = float(getattr(config, "COST_FLIGHT_TIME", 0.1))

        for u, v in self.A_F:
            options = []
            if self.inst.get_dist(u, v) <= config.MAX_DIRECT_DISTANCE:
                options.append("0")
                self.t[u, v, "0"] = self.inst.get_time(u, v)

            for k in self.inst.E_uv.get((u, v), []):
                options.append(k)
                self.t[u, v, k] = self.inst.get_time(u, k) + self.inst.get_time(k, v)

            seen = set()
            ordered_options = []
            for k in options:
                if k not in seen:
                    ordered_options.append(k)
                    seen.add(k)

            self.route_options[u, v] = ordered_options
            for k in ordered_options:
                self.flight_cost[u, v, k] = flight_cost_rate * self.t[u, v, k]

    def _hub_pool(self, h: str) -> str:
        return self.inst.h_to_s[h]

    def _p_chain(self, hub: str) -> float:
        p_by_hub = getattr(config, "P_CHAIN_BY_HUB", None)
        if isinstance(p_by_hub, dict):
            return float(p_by_hub.get(hub, config.P_CHAIN))
        return float(config.P_CHAIN)

    def _demand(self, period: str, i: str, j: str) -> float:
        return float(self.mp.demand.get(period, {}).get((i, j), 0.0))

    def build(self) -> None:
        if self._built:
            return

        C = range(len(config.MU_HUB))
        mu_c = [float(v) for v in config.MU_HUB]
        tau_H = float(config.TAU_H)
        tau_S = float(config.TAU_S)
        omega = float(config.ENERGY_RATE_OMEGA)
        Omega = float(config.BATTERY_CAPACITY_OMEGA_UPPER)
        energy_epsilon = float(getattr(config, "ENERGY_EPSILON", getattr(config, "EPS", 0.0)))
        max_flow = float(config.MAX_FLOW_E)
        usable_battery = (1.0 - energy_epsilon) * Omega

        x = self.model.addVars(self.N_x, vtype=GRB.BINARY, name="x")
        z = self.model.addVars(self.H, C, vtype=GRB.BINARY, name="z")

        y_keys = [(i, l) for (l, i) in self.A_LH]
        y = self.model.addVars(y_keys, vtype=GRB.BINARY, name="y")

        rho_l = self.model.addVars(self.T, self.L, lb=0.0, name="rho_l")
        rho_s = self.model.addVars(self.T, self.S, lb=0.0, name="rho_s")

        alpha_keys = [
            (period, l, i, j)
            for period in self.T
            for (l, i) in self.A_LH
            for (i2, j) in self.A_HH
            if i2 == i
        ]
        alpha = self.model.addVars(alpha_keys, lb=0.0, name="alpha")

        beta_keys = [(period, i, j) for period in self.T for (i, j) in self.A_HH]
        beta = self.model.addVars(beta_keys, lb=0.0, name="beta")

        psi_keys = [(period, u, v) for period in self.T for (u, v) in self.A]
        psi = self.model.addVars(psi_keys, lb=0.0, name="psi")

        varphi_keys = [
            (period, u, v, k)
            for period in self.T
            for (u, v) in self.A_F
            for k in self.route_options.get((u, v), [])
        ]
        varphi = self.model.addVars(varphi_keys, lb=0.0, name="varphi")

        lambd = self.model.addVars(self.T, self.H, lb=0.0, name="lambda")
        xi = self.model.addVars(self.T, self.HLS, lb=0.0, name="xi")
        Gamma = self.model.addVar(lb=0.0, vtype=GRB.INTEGER, name="Gamma")

        served_benefit = gp.quicksum(
            self.period_weights[period]
            * float(getattr(config, "ORDER_REVENUE_COEF", 0.0))
            * psi[period, i, j]
            for period in self.T
            for i, j in self.A_HH
        )
        fixed_cost = gp.quicksum(float(self.inst.f.get(l, config.COST_BASE)) * x[l] for l in self.L)
        fixed_cost += gp.quicksum(float(getattr(config, "COST_HUB_FIXED", 0.0)) * x[i] for i in self.H)
        fixed_cost += gp.quicksum(float(self.inst.f.get(k, config.COST_SWAPPING)) * x[k] for k in self.E)
        fixed_cost += gp.quicksum(float(config.COST_HUB[c]) * z[i, c] for i in self.H for c in C)
        operating_cost = gp.quicksum(
            self.period_weights[period] * self.flight_cost[u, v, k] * varphi[period, u, v, k]
            for period, u, v, k in varphi_keys
        )

        self.model.setObjective(
            served_benefit
            - fixed_cost
            - operating_cost
            - float(config.COST_DRONE) * Gamma,
            GRB.MAXIMIZE,
        )

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

        mu_expr = {
            i: gp.quicksum(mu_c[c] * z[i, c] for c in C)
            for i in self.H
        }
        sqrt_mu_expr = {
            i: gp.quicksum((mu_c[c] ** 0.5) * z[i, c] for c in C)
            for i in self.H
        }

        q_nodes = self.L + self.S
        q_v1 = self.model.addVars(self.T, q_nodes, lb=-GRB.INFINITY, name="q_v1")
        q_v2 = self.model.addVars(self.T, q_nodes, lb=-GRB.INFINITY, name="q_v2")
        q_t = self.model.addVars(self.T, q_nodes, lb=0.0, name="q_t")
        h_v1 = self.model.addVars(self.T, self.H, lb=-GRB.INFINITY, name="h_v1")
        h_v2 = self.model.addVars(self.T, self.H, lb=-GRB.INFINITY, name="h_v2")
        h_t = self.model.addVars(self.T, self.H, lb=0.0, name="h_t")

        for period in self.T:
            for l in self.L:
                self.model.addConstr(rho_l[period, l] <= x[l], name=f"rho_L_{period}_{l}")
            for i, s in self.A_HS:
                self.model.addConstr(rho_s[period, s] <= x[i], name=f"rho_S_{period}_{s}")

            for _, l, i, j in (key for key in alpha_keys if key[0] == period):
                self.model.addConstr(alpha[period, l, i, j] <= rho_l[period, l], name=f"alpha_1_{period}_{l}_{i}_{j}")
                self.model.addConstr(alpha[period, l, i, j] <= y[i, l], name=f"alpha_2_{period}_{l}_{i}_{j}")
                self.model.addConstr(alpha[period, l, i, j] <= x[j], name=f"alpha_3_{period}_{l}_{i}_{j}")
                self.model.addConstr(
                    alpha[period, l, i, j] >= rho_l[period, l] + y[i, l] + x[j] - 2,
                    name=f"alpha_4_{period}_{l}_{i}_{j}",
                )

            for s, i in self.A_SH:
                for h, j in self.A_HH:
                    if h != i:
                        continue
                    self.model.addConstr(beta[period, i, j] <= rho_s[period, s], name=f"beta_1_{period}_{i}_{j}")
                    self.model.addConstr(beta[period, i, j] <= x[j], name=f"beta_2_{period}_{i}_{j}")
                    self.model.addConstr(
                        beta[period, i, j] >= rho_s[period, s] + x[j] - 1,
                        name=f"beta_3_{period}_{i}_{j}",
                    )

            for i, j in self.A_HH:
                p_i = self._p_chain(i)
                demand = self._demand(period, i, j)
                self.model.addConstr(
                    psi[period, i, j]
                    == demand
                    * (
                        (1.0 - p_i)
                        * gp.quicksum(
                            alpha[period, l, i, j]
                            for (l, h) in self.A_LH
                            if h == i and (period, l, i, j) in alpha
                        )
                        + p_i * beta[period, i, j]
                    ),
                    name=f"psi_HH_{period}_{i}_{j}",
                )

            for l, i in self.A_LH:
                self.model.addConstr(
                    psi[period, l, i]
                    == gp.quicksum(
                        self._demand(period, i, j)
                        * (1.0 - self._p_chain(i))
                        * alpha[period, l, i, j]
                        for (h, j) in self.A_HH
                        if h == i and (period, l, i, j) in alpha
                    ),
                    name=f"psi_LH_{period}_{l}_{i}",
                )

            for s, i in self.A_SH:
                self.model.addConstr(
                    psi[period, s, i]
                    == gp.quicksum(
                        self._demand(period, i, j)
                        * self._p_chain(i)
                        * beta[period, i, j]
                        for (h, j) in self.A_HH
                        if h == i
                    ),
                    name=f"psi_SH_{period}_{s}_{i}",
                )

            for q in self.HLS:
                self.model.addConstr(
                    gp.quicksum(psi[period, u, q] for (u, v) in self.A if v == q)
                    == gp.quicksum(psi[period, q, v] for (u, v) in self.A if u == q),
                    name=f"flow_balance_{period}_{q}",
                )

            for i, s in self.A_HS:
                self.model.addConstr(
                    gp.quicksum(psi[period, s, u] for (v, u) in self.A if v == s) <= max_flow * x[i],
                    name=f"pool_flow_active_{period}_{s}",
                )

            for u, v in self.A_F:
                self.model.addConstr(
                    gp.quicksum(
                        varphi[period, u, v, k]
                        for k in self.route_options.get((u, v), [])
                        if (period, u, v, k) in varphi
                    )
                    == psi[period, u, v],
                    name=f"route_{period}_{u}_{v}",
                )

            for k in self.E:
                swap_capacity = float(self.inst.mu.get(k, getattr(config, "SWAP_CAPACITY", config.MU_SWAPPING)))
                self.model.addConstr(
                    gp.quicksum(varphi[period, u, v, kk] for (tp, u, v, kk) in varphi_keys if tp == period and kk == k)
                    <= swap_capacity * x[k],
                    name=f"swap_capacity_{period}_{k}",
                )

            total_queue_length = gp.quicksum(xi[period, u] for u in self.HLS)
            total_flight_time = gp.quicksum(
                self.t[u, v, k] * varphi[period, u, v, k]
                for (tp, u, v, k) in varphi_keys
                if tp == period
            )

            for l in self.L:
                self.model.addConstr(
                    omega
                    * gp.quicksum(
                        self.t[l, i, k] * varphi[period, l, i, k]
                        for (tp, u, i, k) in varphi_keys
                        if tp == period and u == l and (l, i) in self.A_LH
                    )
                    <= usable_battery
                    * gp.quicksum(
                        varphi[period, l, i, k]
                        for (tp, u, i, k) in varphi_keys
                        if tp == period and u == l and (l, i) in self.A_LH and k in self.E
                    ),
                    name=f"energy_L_{period}_{l}",
                )

            for i in self.H:
                s = self._hub_pool(i)
                self.model.addConstr(
                    lambd[period, i]
                    == gp.quicksum(psi[period, u, i] for (u, v) in self.A if v == i),
                    name=f"lambda_{period}_{i}",
                )
                self.model.addConstr(
                    mu_expr[i] - lambd[period, i] >= x[i] / tau_H,
                    name=f"hub_wait_{period}_{i}",
                )
                self.model.addConstr(
                    omega
                    * (
                        xi[period, i]
                        + xi[period, s]
                        + gp.quicksum(
                            self.t[i, j, k] * varphi[period, i, j, k]
                            for (tp, h, j, k) in varphi_keys
                            if tp == period and h == i and (i, j) in self.A_HH
                        )
                        + gp.quicksum(
                            self.t[s, v, k] * varphi[period, s, v, k]
                            for (tp, u, v, k) in varphi_keys
                            if tp == period and u == s and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                        )
                    )
                    <= usable_battery
                    * (
                        gp.quicksum(
                            varphi[period, i, j, k]
                            for (tp, h, j, k) in varphi_keys
                            if tp == period and h == i and (i, j) in self.A_HH and k in self.E
                        )
                        + gp.quicksum(
                            varphi[period, s, v, k]
                            for (tp, u, v, k) in varphi_keys
                            if tp == period
                            and u == s
                            and ((s, v) in self.A_SL or (s, v) in self.A_SS)
                            and k in self.E
                        )
                    ),
                    name=f"energy_HS_{period}_{i}",
                )

            for q in q_nodes:
                rho_q = rho_l[period, q] if q in self.L else rho_s[period, q]
                self.model.addConstr(q_v1[period, q] == 2.0, name=f"q_v1_{period}_{q}")
                self.model.addConstr(q_v2[period, q] == xi[period, q] + rho_q, name=f"q_v2_{period}_{q}")
                self.model.addConstr(q_t[period, q] == xi[period, q] - rho_q + 2.0, name=f"q_t_{period}_{q}")
                q_name = f"queue_L_{period}_{q}" if q in self.L else f"queue_S_{period}_{q}"
                self.model.addQConstr(
                    q_v1[period, q] * q_v1[period, q] + q_v2[period, q] * q_v2[period, q]
                    <= q_t[period, q] * q_t[period, q],
                    name=q_name,
                )

            for i in self.H:
                self.model.addConstr(h_v1[period, i] == 2.0 * sqrt_mu_expr[i], name=f"h_v1_{period}_{i}")
                self.model.addConstr(
                    h_v2[period, i] == xi[period, i] + lambd[period, i] - mu_expr[i] + 1.0,
                    name=f"h_v2_{period}_{i}",
                )
                self.model.addConstr(
                    h_t[period, i] == xi[period, i] - lambd[period, i] + mu_expr[i] + 1.0,
                    name=f"h_t_{period}_{i}",
                )
                self.model.addQConstr(
                    h_v1[period, i] * h_v1[period, i] + h_v2[period, i] * h_v2[period, i]
                    <= h_t[period, i] * h_t[period, i],
                    name=f"queue_H_{period}_{i}",
                )

            for s in self.S:
                self.model.addConstr(
                    xi[period, s]
                    <= tau_S * gp.quicksum(psi[period, s, v] for (u, v) in self.A if u == s),
                    name=f"mean_S_{period}_{s}",
                )

            self.model.addConstr(
                total_queue_length + total_flight_time <= Gamma,
                name=f"fleet_{period}",
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
            "Gamma": Gamma,
            "q_v1": q_v1,
            "q_v2": q_v2,
            "q_t": q_t,
            "h_v1": h_v1,
            "h_v2": h_v2,
            "h_t": h_t,
        }
        self._built = True

    def apply_solver_config(self) -> None:
        time_limit = getattr(config, "MODEL_TIME_LIMIT", None)
        mip_gap = getattr(config, "MODEL_MIP_GAP", None)
        if time_limit is not None:
            self.model.setParam("TimeLimit", float(time_limit))
        if mip_gap is not None:
            self.model.setParam("MIPGap", float(mip_gap))

    def solve(self) -> bool:
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

    def write_model(self, filename="multi_period_model.lp") -> None:
        self.model.write(filename)


def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "n"}


def _hour_from_period(period_id: str) -> int:
    if not period_id.startswith("h"):
        raise ValueError(f"Cannot infer hour from period id {period_id!r}.")
    return int(period_id[1:3])


def _write_period_manifest(instance_dir: str, periods: Iterable[str], weights: Dict[str, float]) -> None:
    rows = []
    for period_id in periods:
        hour = _hour_from_period(period_id)
        rows.append(
            {
                "period_id": period_id,
                "orders_file": os.path.join("orders", f"{period_id}.csv"),
                "start_hour": hour,
                "end_hour": hour + 1,
                "hour_semantics": "[start, end)",
                "weight": weights.get(period_id, 1.0),
            }
        )

    with open(os.path.join(instance_dir, "periods.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["period_id", "orders_file", "start_hour", "end_hour", "hour_semantics", "weight"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_meituan_regime_ii_multi_period(
    *,
    generate_sample=None,
    date_filter=None,
    project_root=None,
    instance_dir=None,
    demand_hours=None,
    period_ids=None,
    period_weights=None,
    num_hubs=None,
    num_bases=None,
    num_swapping=None,
    write_lp=False,
) -> MultiPeriodMLNDModel:
    """Generate/load a Meituan multi-period instance and solve the shared-design model."""
    from src.data_processor.generator import DataGenerator

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if generate_sample is None:
        generate_sample = _env_flag("MLND_GENERATE_SAMPLE", True)
    if date_filter is None:
        date_filter = config.MEITUAN_DEMAND_DATES[0]

    periods = _as_period_ids(period_ids=period_ids, demand_hours=demand_hours)
    weights = _weights_by_period(periods, period_weights)

    if instance_dir is None:
        instance_dir = os.path.join(project_root, "data", "instances", "MeituanMultiPeriod")
    raw_file = os.path.join(project_root, "Meituan-TSL", "all_waybill_info_meituan_0322.csv")

    if num_hubs is None:
        num_hubs = config.MEITUAN_NUM_HUBS
    if num_bases is None:
        num_bases = config.MEITUAN_NUM_BASES
    if num_swapping is None:
        num_swapping = config.MEITUAN_NUM_SWAPPING

    print(
        f"Regime II multi-period run: generate={generate_sample}, periods={periods}, "
        f"weights={weights}, H={num_hubs}, L={num_bases}, E={num_swapping}"
    )

    if generate_sample:
        generator = DataGenerator(seed=42)
        for period_id in periods:
            hour = _hour_from_period(period_id)
            generator.generate_meituan_hourly(
                raw_data_path=raw_file,
                save_dir=instance_dir,
                num_hubs=num_hubs,
                num_bases=num_bases,
                num_swapping=num_swapping,
                date_filter=date_filter,
                map_half_size_km=config.MEITUAN_MAP_HALF_SIZE_KM,
                candidate_date_filter=config.MEITUAN_CANDIDATE_DATES,
                demand_date_filter=config.MEITUAN_DEMAND_DATES,
                candidate_hour_start=config.MEITUAN_CANDIDATE_HOUR_START,
                candidate_hour_end=config.MEITUAN_CANDIDATE_HOUR_END,
                demand_hour_start=hour,
                demand_hour_end=hour + 1,
                colocate_base_swaps=config.MEITUAN_COLOCATE_BASE_SWAPS,
            )
        _write_period_manifest(instance_dir, periods, weights)

    mp_inst = load_multi_period_instance(instance_dir, periods, period_weights=weights)
    print(
        f"Loaded multi-period: T={len(mp_inst.periods)}, H={len(mp_inst.base.H)}, "
        f"L={len(mp_inst.base.L)}, S={len(mp_inst.base.S)}, E={len(mp_inst.base.E)}, "
        f"OD_union={len(mp_inst.base.h)}, A={len(mp_inst.base.A)}, A_F={len(mp_inst.base.A_F)}"
    )
    for period_id in mp_inst.periods:
        total_demand = sum(mp_inst.demand[period_id].values())
        print(f"  {period_id}: OD={len(mp_inst.demand[period_id])}, demand={total_demand:.4f}/h")

    model = MultiPeriodMLNDModel(mp_inst)
    success = model.solve()
    if write_lp:
        results_dir = os.path.join(project_root, "results", "MeituanMultiPeriod")
        os.makedirs(results_dir, exist_ok=True)
        model.write_model(os.path.join(results_dir, "model_regime_ii_multi_period.lp"))
        print(f"Saved LP to {results_dir}")
    if not success:
        print("No feasible solution found.")
    return model


if __name__ == "__main__":
    run_meituan_regime_ii_multi_period()
