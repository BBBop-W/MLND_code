"""Gurobi implementation of the new OD-aware mixed-delivery MISOCP."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from src.config import SolverConfig
from src.data_processor.instance import DIRECT_ROUTE, Arc, Instance, Market, OrderCorridor
from src.model.primal_heuristic import (
    LocalBranchingConfig,
    capture_incumbent,
    improve_mip_start,
)
from src.model.valid_inequalities import (
    ChoiceCorridorCoverData,
    build_choice_corridor_cover_data,
    queue_perspective_intercept,
    separate_choice_capacity_covers,
    separate_directed_return_cuts,
    separate_queue_perspective_cut,
    stable_arrival_capacity,
)


class MLNDModel:
    """Implement M^MX, M^HD, or M^LO with endogenous OD-to-corridor demand."""

    MODES = {"MX", "HD", "LO"}

    def __init__(self, instance: Instance, mode: str = "MX"):
        self.inst = instance
        self.mode = mode.upper()
        if self.mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.model = gp.Model(f"MLND_{self.mode}_OD_AWARE")
        self.solver_config: SolverConfig | None = None
        self.vars: dict[str, object] = {}
        self.cost_expressions: dict[str, gp.LinExpr] = {}
        self.network_start_runtime = 0.0
        self.network_start_found = False
        self.network_start_used = False
        self.network_start_key: tuple[str, str] | None = None
        self.network_start_objective: float | None = None
        self.main_solve_runtime = 0.0
        self.total_runtime = 0.0
        self.static_cut_counts: dict[str, int] = {}
        self.user_cut_counts: dict[str, int] = {}
        self.root_cut_rounds = 0
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective: float | None = None
        self.local_branching_status: int | None = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance: int | None = None
        self._valid_inequalities_added = False
        self._user_cut_signatures: set[tuple[object, ...]] = set()
        self._choice_cover_data: tuple[ChoiceCorridorCoverData, ...] = ()
        self._built = False
        self._prepare_indices()

    def _prepare_indices(self) -> None:
        self.locker_keys: tuple[OrderCorridor, ...] = (
            tuple(self.inst.locker_keys) if self.mode in {"MX", "LO"} else ()
        )
        self.home_keys: tuple[OrderCorridor, ...] = (
            tuple(self.inst.home_keys) if self.mode in {"MX", "HD"} else ()
        )
        self.alpha_keys = tuple(
            (destination, locker)
            for destination in sorted(self.inst.display_lockers)
            for locker in self.inst.display_lockers[destination]
        ) if self.mode in {"MX", "LO"} else ()

        self.lk_by_market_j: dict[tuple[str, str, str], list[OrderCorridor]] = defaultdict(list)
        self.lk_by_market: dict[Market, list[OrderCorridor]] = defaultdict(list)
        self.hm_by_market: dict[Market, list[OrderCorridor]] = defaultdict(list)
        self.offered_by_corridor: dict[Arc, list[tuple[str, OrderCorridor]]] = defaultdict(list)
        self.served_by_corridor: dict[Arc, list[tuple[str, OrderCorridor]]] = defaultdict(list)
        for key in self.locker_keys:
            o, i, j, d = key
            self.lk_by_market_j[o, d, j].append(key)
            self.lk_by_market[o, d].append(key)
            self.offered_by_corridor[i, j].append(("lk", key))
            self.served_by_corridor[i, j].append(("lk", key))
        for key in self.home_keys:
            o, i, j, d = key
            self.hm_by_market[o, d].append(key)
            self.offered_by_corridor[i, j].append(("hm", key))
            self.served_by_corridor[i, j].append(("hm", key))

        self.corridors = tuple(sorted(self.offered_by_corridor))
        self.M_w: dict[Arc, float] = {arc: 0.0 for arc in self.corridors}
        for market in self.inst.markets:
            market_arcs = {
                (key[1], key[2])
                for key in list(self.lk_by_market.get(market, ()))
                + list(self.hm_by_market.get(market, ()))
            }
            for arc in market_arcs:
                self.M_w[arc] += self.inst.h_od[market]
        # Since every offered order is realized at one of the strictly
        # positive rho levels, psi_ij >= rho_min h_ij.  The locker-throughput
        # bound on psi therefore also gives an exact, often much smaller,
        # offered-flow bound for each corridor.
        rho_min = min(self.inst.config.availability_levels)
        physical_bound = max(self.inst.M_psi.values(), default=0.0)
        offered_flow_bound = physical_bound / rho_min
        self.M_w = {
            arc: max(0.0, min(value, offered_flow_bound))
            for arc, value in self.M_w.items()
        }

        self.w_keys: list[tuple[str, str, str, int]] = []
        self.w_by_source: dict[tuple[str, str], list[tuple[str, str, str, int]]] = defaultdict(list)
        self.w_by_corridor: dict[Arc, list[tuple[str, str, str, int]]] = defaultdict(list)
        for i, j in self.corridors:
            for idle in self.inst.eligible_idle[i]:
                for r in range(len(self.inst.config.availability_levels)):
                    key = (idle, i, j, r)
                    self.w_keys.append(key)
                    self.w_by_source[idle, i].append(key)
                    self.w_by_corridor[i, j].append(key)

        self.varphi_keys = tuple(
            (u, v, option)
            for u, v in sorted(self.inst.A_F)
            for option in self.inst.route_options[u, v]
        )

    def build(self) -> "MLNDModel":
        if self._built:
            return self
        cfg = self.inst.config
        C = range(len(cfg.locker_service_rates))
        R = range(len(cfg.availability_levels))
        physical_nodes = self.inst.L + self.inst.H + self.inst.E
        balance_nodes = self.inst.L + self.inst.H + self.inst.S

        x = self.model.addVars(physical_nodes, vtype=GRB.BINARY, name="x")
        z = self.model.addVars(self.inst.H, C, vtype=GRB.BINARY, name="z")
        y = self.model.addVars(sorted(self.inst.A_Y), vtype=GRB.BINARY, name="y")
        delta = self.model.addVars(
            [(idle, r) for idle in self.inst.idle_states for r in R],
            vtype=GRB.BINARY,
            name="delta",
        )
        alpha = self.model.addVars(self.alpha_keys, vtype=GRB.BINARY, name="alpha")
        gamma = self.model.addVar(lb=0.0, vtype=GRB.INTEGER, name="gamma")

        h = self.model.addVars(self.corridors, lb=0.0, name="h_ij")
        w = self.model.addVars(self.w_keys, lb=0.0, name="w")
        psi = self.model.addVars(sorted(self.inst.A), lb=0.0, name="psi")
        varphi = self.model.addVars(self.varphi_keys, lb=0.0, name="varphi")
        lambd = self.model.addVars(self.inst.H, lb=0.0, name="lambda")
        xi = self.model.addVars(balance_nodes, lb=0.0, name="xi")
        a_lk = self.model.addVars(self.locker_keys, lb=0.0, name="a_lk")
        a_hm = self.model.addVars(self.home_keys, lb=0.0, name="a_hm")
        q_lk = self.model.addVars(self.locker_keys, lb=0.0, name="q_lk")
        q_hm = self.model.addVars(self.home_keys, lb=0.0, name="q_hm")

        facility_cost = gp.quicksum(self.inst.facility_cost(node) * x[node] for node in physical_nodes)
        capacity_cost = gp.quicksum(
            cfg.locker_capacity_costs[c] * z[i, c] for i in self.inst.H for c in C
        )
        connection_cost = gp.quicksum(
            self.inst.arc_activation_cost[arc] * y[arc] for arc in sorted(self.inst.A_Y)
        )
        flight_cost = gp.quicksum(
            cfg.flight_cost_per_km * self.inst.route_distance[key] * varphi[key]
            for key in self.varphi_keys
        )
        fleet_cost = cfg.drone_cost_per_hour * gamma
        locker_fulfillment_cost = gp.quicksum(
            (self.inst.rider_cost(o, i) + cfg.locker_service_cost) * q_lk[o, i, j, d]
            for o, i, j, d in self.locker_keys
        )
        assisted_home_cost = gp.quicksum(
            (self.inst.rider_cost(o, i) + self.inst.rider_cost(j, d)) * q_hm[o, i, j, d]
            for o, i, j, d in self.home_keys
        )
        fallback_cost = gp.LinExpr()
        for market in self.inst.markets:
            o, d = market
            served = gp.quicksum(q_lk[key] for key in self.lk_by_market.get(market, ()))
            served += gp.quicksum(q_hm[key] for key in self.hm_by_market.get(market, ()))
            fallback_cost += self.inst.rider_cost(o, d) * (self.inst.h_od[market] - served)

        self.cost_expressions = {
            "facility": facility_cost,
            "capacity": capacity_cost,
            "connection": connection_cost,
            "flight": flight_cost,
            "fleet": fleet_cost,
            "locker_fulfillment": locker_fulfillment_cost,
            "assisted_home": assisted_home_cost,
            "fallback_home": fallback_cost,
        }
        total_cost = gp.quicksum(self.cost_expressions.values())
        self.courier_only_baseline = sum(
            self.inst.rider_cost(o, d) * self.inst.h_od[o, d] for o, d in self.inst.markets
        )
        self.model.setObjective(total_cost, GRB.MINIMIZE)

        for i in self.inst.H:
            self.model.addConstr(gp.quicksum(z[i, c] for c in C) == x[i], name=f"capacity[{i}]")
        for u, v in sorted(self.inst.A_Y):
            self.model.addConstr(y[u, v] <= x[u], name=f"arc_tail[{u},{v}]")
            self.model.addConstr(y[u, v] <= x[v], name=f"arc_head[{u},{v}]")

        for i, j in self.corridors:
            upper = self.M_w[i, j]
            self.model.addConstr(h[i, j] <= upper * y[i, j], name=f"demand_open[{i},{j}]")
            self.model.addConstr(
                gp.quicksum(w[key] for key in self.w_by_corridor[i, j]) == h[i, j],
                name=f"demand_allocate[{i},{j}]",
            )

        for base in self.inst.L:
            self.model.addConstr(
                gp.quicksum(delta[base, r] for r in R) == x[base],
                name=f"availability_base[{base}]",
            )
        for state in self.inst.S:
            self.model.addConstr(
                gp.quicksum(delta[state, r] for r in R) <= 1,
                name=f"availability_virtual[{state}]",
            )
        for idle in self.inst.idle_states:
            self.model.addConstr(
                xi[idle]
                == gp.quicksum(
                    cfg.availability_levels[r] / (1.0 - cfg.availability_levels[r]) * delta[idle, r]
                    for r in R
                ),
                name=f"idle_population[{idle}]",
            )

        for idle, i, j, r in self.w_keys:
            upper = self.M_w[i, j]
            self.model.addConstr(w[idle, i, j, r] <= upper * delta[idle, r])
            if idle in self.inst.L:
                self.model.addConstr(w[idle, i, j, r] <= upper * y[idle, i])
            self.model.addConstr(w[idle, i, j, r] <= upper * y[i, j])

        for i in self.inst.H:
            for idle in self.inst.eligible_idle[i]:
                keys = self.w_by_source.get((idle, i), ())
                self.model.addConstr(
                    psi[idle, i]
                    == gp.quicksum(cfg.availability_levels[key[3]] * w[key] for key in keys),
                    name=f"source_flow[{idle},{i}]",
                )
        for i, j in sorted(self.inst.A_HH):
            keys = self.w_by_corridor.get((i, j), ())
            self.model.addConstr(
                psi[i, j]
                == gp.quicksum(cfg.availability_levels[key[3]] * w[key] for key in keys),
                name=f"loaded_flow[{i},{j}]",
            )

        for node in balance_nodes:
            self.model.addConstr(
                gp.quicksum(psi[u, v] for u, v in self.inst.A if v == node)
                == gp.quicksum(psi[u, v] for u, v in self.inst.A if u == node),
                name=f"flow_balance[{node}]",
            )
        for arc in sorted(self.inst.A_F):
            self.model.addConstr(
                psi[arc] <= self.inst.M_psi[arc] * y[arc], name=f"physical_flow[{arc[0]},{arc[1]}]"
            )

        for u, v in sorted(self.inst.A_F):
            self.model.addConstr(
                gp.quicksum(varphi[u, v, option] for option in self.inst.route_options[u, v]) == psi[u, v],
                name=f"route_decomposition[{u},{v}]",
            )
            for option in self.inst.route_options[u, v]:
                upper = self.inst.M_psi[u, v]
                self.model.addConstr(varphi[u, v, option] <= upper * y[u, v])
                if option != DIRECT_ROUTE:
                    self.model.addConstr(varphi[u, v, option] <= upper * y[u, option])
                    self.model.addConstr(varphi[u, v, option] <= upper * y[option, v])

        q_v1 = self.model.addVars(self.inst.H, lb=-GRB.INFINITY, name="queue_v1")
        q_v2 = self.model.addVars(self.inst.H, lb=-GRB.INFINITY, name="queue_v2")
        q_t = self.model.addVars(self.inst.H, lb=0.0, name="queue_t")
        for i in self.inst.H:
            incoming = gp.quicksum(psi[u, v] for u, v in self.inst.A if v == i)
            mu = gp.quicksum(cfg.locker_service_rates[c] * z[i, c] for c in C)
            self.model.addConstr(lambd[i] == incoming, name=f"locker_workload[{i}]")
            self.model.addConstr(
                mu - lambd[i] >= x[i] / cfg.max_locker_time_hours,
                name=f"locker_qos[{i}]",
            )
            self.model.addConstr(xi[i] <= self.inst.M_xi[i] * x[i], name=f"locker_population[{i}]")
            self.model.addConstr(
                q_v1[i] == 2.0 * gp.quicksum(math.sqrt(cfg.locker_service_rates[c]) * z[i, c] for c in C)
            )
            self.model.addConstr(q_v2[i] == xi[i] + lambd[i] - mu + 1.0)
            self.model.addConstr(q_t[i] == xi[i] - lambd[i] + mu + 1.0)
            self.model.addQConstr(
                q_v1[i] * q_v1[i] + q_v2[i] * q_v2[i] <= q_t[i] * q_t[i],
                name=f"locker_queue_soc[{i}]",
            )
        for state in self.inst.S:
            outflow = gp.quicksum(psi[u, v] for u, v in self.inst.A if u == state)
            self.model.addConstr(
                xi[state] <= cfg.max_virtual_wait_hours * outflow,
                name=f"holding_qos[{state}]",
            )
        self.model.addConstr(
            gp.quicksum(xi[node] for node in balance_nodes)
            + gp.quicksum(self.inst.route_time[key] * varphi[key] for key in self.varphi_keys)
            <= gamma,
            name="fleet_size",
        )

        if self.mode in {"MX", "LO"}:
            for destination in sorted(self.inst.display_lockers):
                lockers = self.inst.display_lockers[destination]
                if not lockers:
                    continue
                self.model.addConstr(
                    gp.quicksum(alpha[destination, j] for j in lockers) <= 1,
                    name=f"display_limit[{destination}]",
                )
                for j in lockers:
                    self.model.addConstr(alpha[destination, j] <= x[j])
                    for closer in self.inst.closer_lockers[destination, j]:
                        self.model.addConstr(alpha[destination, j] <= 1 - x[closer])
            for market in self.inst.markets:
                o, d = market
                for j in self.inst.display_lockers.get(d, ()):
                    keys = self.lk_by_market_j[o, d, j]
                    self.model.addConstr(
                        gp.quicksum(a_lk[key] for key in keys)
                        == self.inst.h_od[market] * self.inst.choice_share[o, d, j] * alpha[d, j],
                        name=f"locker_offer[{o},{d},{j}]",
                    )

        if self.mode in {"MX", "HD"}:
            for market in self.inst.markets:
                locker_offer = gp.quicksum(a_lk[key] for key in self.lk_by_market.get(market, ()))
                self.model.addConstr(
                    gp.quicksum(a_hm[key] for key in self.hm_by_market[market])
                    <= self.inst.h_od[market] - locker_offer,
                    name=f"home_offer[{market[0]},{market[1]}]",
                )

        for i, j in self.corridors:
            terms = []
            for kind, key in self.offered_by_corridor[i, j]:
                terms.append(a_lk[key] if kind == "lk" else a_hm[key])
            self.model.addConstr(h[i, j] == gp.quicksum(terms), name=f"corridor_offer[{i},{j}]")
        for key in self.locker_keys:
            self.model.addConstr(q_lk[key] <= a_lk[key])
        for key in self.home_keys:
            self.model.addConstr(q_hm[key] <= a_hm[key])
        for i, j in sorted(self.inst.A_HH):
            terms = []
            for kind, key in self.served_by_corridor.get((i, j), ()):
                terms.append(q_lk[key] if kind == "lk" else q_hm[key])
            self.model.addConstr(gp.quicksum(terms) == psi[i, j], name=f"served_flow[{i},{j}]")

        self.vars = {
            "x": x,
            "z": z,
            "y": y,
            "delta": delta,
            "alpha": alpha,
            "gamma": gamma,
            "h": h,
            "w": w,
            "psi": psi,
            "varphi": varphi,
            "lambda": lambd,
            "xi": xi,
            "a_lk": a_lk,
            "a_hm": a_hm,
            "q_lk": q_lk,
            "q_hm": q_hm,
            "queue_v1": q_v1,
            "queue_v2": q_v2,
            "queue_t": q_t,
        }
        self._built = True
        self.model.update()
        return self

    def prepare_algorithm(self, solver: SolverConfig | None = None) -> "MLNDModel":
        """Add globally valid strengthening rows before branch-and-cut starts."""

        self.build()
        solver = solver or SolverConfig()
        solver.validate()
        if self.alpha_keys and not self._choice_cover_data:
            self._choice_cover_data = build_choice_corridor_cover_data(
                self.inst.h_od,
                {
                    key: self.inst.choice_share[key]
                    for key in self.inst.locker_origins
                    if (key[1], key[2]) in self.alpha_keys
                },
                self.inst.locker_origins,
            )
        if solver.use_valid_inequalities and not self._valid_inequalities_added:
            self._add_static_valid_inequalities()
            self._valid_inequalities_added = True
            self.model.update()
        return self

    def _add_static_valid_inequalities(self) -> None:
        """Add compact projections of the choice, reliability, and route blocks."""

        cfg = self.inst.config
        x = self.vars["x"]
        z = self.vars["z"]
        y = self.vars["y"]
        delta = self.vars["delta"]
        alpha = self.vars["alpha"]
        w = self.vars["w"]
        varphi = self.vars["varphi"]
        lambd = self.vars["lambda"]
        xi = self.vars["xi"]
        counts: dict[str, int] = defaultdict(int)
        rho_min = min(cfg.availability_levels)
        rho_max = max(cfg.availability_levels)
        max_locker_flow = max(
            0.0,
            min(
                rho_max * self.inst.total_demand,
                max(
                    rate - 1.0 / cfg.max_locker_time_hours
                    for rate in cfg.locker_service_rates
                ),
            ),
        )

        # A virtual state co-located with locker i cannot be active when i is
        # closed.  This is implied only after several flow/QoS rows at integer
        # points, so the explicit binary link strengthens the relaxation.
        for locker in self.inst.H:
            state = self.inst.h_to_s[locker]
            self.model.addConstr(
                gp.quicksum(delta[state, r] for r in range(len(cfg.availability_levels)))
                <= x[locker],
                name=f"vi_virtual_open[{locker}]",
            )
            counts["virtual_open_link"] += 1

        # Convex-hull bounds for one selected reliability level.  The upper
        # bound follows from locker throughput; virtual-state lower bounds are
        # induced by xi_s <= tau^S psi_s and xi_s=rho/(1-rho).
        for locker in self.inst.H:
            for idle in self.inst.eligible_idle[locker]:
                keys = self.w_by_source.get((idle, locker), ())
                for r, rho in enumerate(cfg.availability_levels):
                    level_keys = [key for key in keys if key[3] == r]
                    offered = gp.quicksum(w[key] for key in level_keys)
                    self.model.addConstr(
                        rho * offered <= max_locker_flow * delta[idle, r],
                        name=f"vi_source_level_upper[{idle},{locker},{r}]",
                    )
                    counts["source_level_upper"] += 1
                    if idle in self.inst.S:
                        lower = 1.0 / (
                            (1.0 - rho) * cfg.max_virtual_wait_hours
                        )
                        self.model.addConstr(
                            offered >= lower * delta[idle, r],
                            name=f"vi_virtual_level_lower[{idle},{r}]",
                        )
                        counts["virtual_level_lower"] += 1

        # Displaying locker j creates a fixed amount of potential locker
        # demand.  Activated backend corridors must provide enough aggregate
        # offered-flow capacity to carry it.
        display_volume: dict[tuple[str, str], float] = {}
        display_arcs: dict[tuple[str, str], set[Arc]] = defaultdict(set)
        for destination, locker in self.alpha_keys:
            volume = 0.0
            for market in self.inst.markets:
                origin, d_node = market
                if d_node != destination:
                    continue
                volume += (
                    self.inst.h_od[market]
                    * self.inst.choice_share[origin, destination, locker]
                )
                for key in self.lk_by_market_j.get((origin, destination, locker), ()):
                    display_arcs[destination, locker].add((key[1], key[2]))
            display_volume[destination, locker] = volume
            if volume <= 0.0:
                continue
            arcs = display_arcs[destination, locker]
            self.model.addConstr(
                gp.quicksum(
                    min(self.M_w[arc], volume) * y[arc]
                    for arc in arcs
                )
                >= volume * alpha[destination, locker],
                name=f"vi_choice_corridor_cover[{destination},{locker}]",
            )
            counts["choice_corridor_cover"] += 1

        # Projection from displayed demand through rho_min and the locker QoS
        # capacity.  Lifted cover cuts for this knapsack are separated at root.
        capacity_slack = tuple(
            rate - 1.0 / cfg.max_locker_time_hours
            for rate in cfg.locker_service_rates
        )
        for locker in self.inst.H:
            forced = gp.quicksum(
                display_volume[destination, candidate]
                * alpha[destination, candidate]
                for destination, candidate in self.alpha_keys
                if candidate == locker
            )
            self.model.addConstr(
                rho_min * forced
                <= gp.quicksum(
                    capacity_slack[c] * z[locker, c]
                    for c in range(len(capacity_slack))
                ),
                name=f"vi_choice_capacity[{locker}]",
            )
            counts["choice_capacity_projection"] += 1

        # A small deterministic support grid gives the root relaxation the
        # convex-hull geometry of the discrete M/M/1 capacity disjunction
        # before callback separation starts.  Dynamic separation later adds
        # the most violated common slope at the actual fractional point.
        queue_slopes = sorted(
            {
                slope
                for rate, slack in zip(cfg.locker_service_rates, capacity_slack)
                for slope in (
                    1.0 / rate,
                    rate / (rate - max(0.0, slack)) ** 2,
                )
                if slope > 0.0 and math.isfinite(slope)
            }
        )
        for locker in self.inst.H:
            for slope_index, slope in enumerate(queue_slopes):
                intercepts = tuple(
                    queue_perspective_intercept(
                        rate,
                        stable_arrival_capacity(rate, cfg.max_locker_time_hours),
                        slope,
                    )
                    for rate in cfg.locker_service_rates
                )
                self.model.addConstr(
                    xi[locker]
                    >= slope * lambd[locker]
                    + gp.quicksum(
                        intercepts[c] * z[locker, c]
                        for c in range(len(intercepts))
                    ),
                    name=f"vi_queue_perspective[{locker},{slope_index}]",
                )
                counts["queue_perspective_support"] += 1

        # Shared swap legs and stations are fixed-charge resources.  Total
        # physical movement is at most three times realized loaded flow: one
        # loaded movement, one base dispatch, and one base recovery.
        total_route_bound = 3.0 * rho_max * self.inst.total_demand
        swap_keys_by_station: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        swap_keys_by_leg: dict[Arc, list[tuple[str, str, str]]] = defaultdict(list)
        for key in self.varphi_keys:
            u, v, option = key
            if option == DIRECT_ROUTE:
                continue
            swap_keys_by_station[option].append(key)
            swap_keys_by_leg[u, option].append(key)
            swap_keys_by_leg[option, v].append(key)
        for station, keys in swap_keys_by_station.items():
            self.model.addConstr(
                gp.quicksum(varphi[key] for key in keys)
                <= total_route_bound * x[station],
                name=f"vi_swap_station_flow[{station}]",
            )
            counts["swap_station_flow"] += 1
        for (u, v), keys in swap_keys_by_leg.items():
            physical_endpoint = u if u not in self.inst.E else v
            multiplier = 1.0 if physical_endpoint in self.inst.L else 2.0
            self.model.addConstr(
                gp.quicksum(varphi[key] for key in keys)
                <= multiplier * rho_max * self.inst.total_demand * y[u, v],
                name=f"vi_swap_leg_flow[{u},{v}]",
            )
            counts["swap_leg_flow"] += 1

        self.static_cut_counts = dict(sorted(counts.items()))

    @staticmethod
    def _callback_values(
        callback_model: gp.Model,
        container: gp.tupledict,
    ) -> dict[object, float]:
        """Read one tupledict from a MIPNODE relaxation in one bulk call."""

        items = list(container.items())
        if not items:
            return {}
        values = callback_model.cbGetNodeRel([variable for _, variable in items])
        return {key: float(value) for (key, _), value in zip(items, values)}

    def _add_root_user_cuts(
        self,
        callback_model: gp.Model,
        solver: SolverConfig,
    ) -> int:
        """Separate one bounded round of globally valid root-node cuts."""

        tolerance = solver.cut_violation_tolerance
        budget = solver.max_cuts_per_round
        if budget <= 0:
            return 0

        cfg = self.inst.config
        a_lk = self.vars["a_lk"]
        a_hm = self.vars["a_hm"]
        alpha = self.vars["alpha"]
        h = self.vars["h"]
        y = self.vars["y"]
        z = self.vars["z"]
        lambd = self.vars["lambda"]
        xi = self.vars["xi"]

        y_values = self._callback_values(callback_model, y)
        counts: dict[str, int] = defaultdict(int)
        added = 0

        # OD-disaggregated variable-upper-bound cuts.  These are much tighter
        # than routing a small fractional share of y through a corridor-level
        # big-M row, but are generated only when violated to avoid another
        # six-figure static row block.
        vub_candidates: list[
            tuple[float, tuple[object, ...], gp.Var, float, gp.Var, str]
        ] = []
        if self.locker_keys:
            locker_values = self._callback_values(callback_model, a_lk)
            for key, value in locker_values.items():
                o, i, j, d = key
                coefficient = self.inst.h_od[o, d] * self.inst.choice_share[o, d, j]
                violation = value - coefficient * y_values[i, j]
                signature = ("od_vub_locker", key)
                if violation > tolerance and signature not in self._user_cut_signatures:
                    vub_candidates.append(
                        (violation, signature, a_lk[key], coefficient, y[i, j], "od_vub_locker")
                    )
        if self.home_keys:
            home_values = self._callback_values(callback_model, a_hm)
            for key, value in home_values.items():
                o, i, j, d = key
                coefficient = self.inst.h_od[o, d]
                violation = value - coefficient * y_values[i, j]
                signature = ("od_vub_home", key)
                if violation > tolerance and signature not in self._user_cut_signatures:
                    vub_candidates.append(
                        (violation, signature, a_hm[key], coefficient, y[i, j], "od_vub_home")
                    )
        vub_candidates.sort(key=lambda item: (-item[0], item[1]))
        vub_quota = max(1, budget // 2)
        for _, signature, offer, coefficient, corridor_open, family in vub_candidates[:vub_quota]:
            callback_model.cbCut(offer <= coefficient * corridor_open)
            self._user_cut_signatures.add(signature)
            counts[family] += 1
            added += 1

        # Most-violated support of the discrete-capacity queue union.
        remaining = budget - added
        queue_quota = min(remaining, max(1, budget // 6))
        queue_added = 0
        if queue_quota > 0:
            lambda_values = self._callback_values(callback_model, lambd)
            xi_values = self._callback_values(callback_model, xi)
            z_values = self._callback_values(callback_model, z)
            for locker in self.inst.H:
                if queue_added >= queue_quota:
                    break
                capacity_values = tuple(
                    z_values[locker, c]
                    for c in range(len(cfg.locker_service_rates))
                )
                weighted_capacity = sum(
                    stable_arrival_capacity(rate, cfg.max_locker_time_hours) * capacity_values[c]
                    for c, rate in enumerate(cfg.locker_service_rates)
                )
                if lambda_values[locker] > weighted_capacity + 10.0 * tolerance:
                    continue
                cut = separate_queue_perspective_cut(
                    lambda_values[locker],
                    xi_values[locker],
                    capacity_values,
                    cfg.locker_service_rates,
                    cfg.max_locker_time_hours,
                    violation_tolerance=tolerance,
                    feasibility_tolerance=max(1e-9, 10.0 * tolerance),
                )
                if cut is None:
                    continue
                signature = ("queue", locker, *cut.signature())
                if signature in self._user_cut_signatures:
                    continue
                callback_model.cbCut(
                    xi[locker]
                    >= cut.lambda_coefficient * lambd[locker]
                    + gp.quicksum(
                        cut.z_coefficients[c] * z[locker, c]
                        for c in range(len(cut.z_coefficients))
                    )
                )
                self._user_cut_signatures.add(signature)
                counts["queue_perspective"] += 1
                added += 1
                queue_added += 1

        # Lifted covers for the display-choice/capacity knapsack.
        if added < budget and self._choice_cover_data:
            alpha_values = self._callback_values(callback_model, alpha)
            z_values = self._callback_values(callback_model, z)
            cover_cuts = separate_choice_capacity_covers(
                self._choice_cover_data,
                tuple(
                    stable_arrival_capacity(rate, cfg.max_locker_time_hours)
                    for rate in cfg.locker_service_rates
                ),
                min(cfg.availability_levels),
                alpha_values,
                z_values,
                violation_tolerance=tolerance,
                feasibility_tolerance=max(1e-8, 10.0 * tolerance),
                max_cuts_per_locker=2,
                seen_signatures=self._user_cut_signatures,
            )
            cover_quota = min(budget - added, max(1, budget // 5))
            for cut in cover_cuts[:cover_quota]:
                if added >= budget:
                    break
                callback_model.cbCut(
                    gp.quicksum(alpha[d, cut.locker] for d in cut.destinations)
                    <= len(cut.destinations)
                    - 1
                    + gp.quicksum(
                        z[cut.locker, c] for c in cut.adequate_capacity_indices
                    )
                )
                self._user_cut_signatures.add(cut.signature())
                counts["choice_capacity_cover"] += 1
                added += 1

        # Conditional connectivity: only positive endogenous h_ij requires a
        # return path.  This deliberately avoids the invalid historical cut
        # that imposed connectivity for every exogenous OD market.
        if added < budget:
            h_values = self._callback_values(callback_model, h)
            return_cuts = separate_directed_return_cuts(
                self.inst.A_F,
                {arc: y_values[arc] for arc in self.inst.A_F},
                h_values,
                self.M_w,
                contraction=self.inst.s_to_h,
                violation_tolerance=tolerance,
                feasibility_tolerance=max(1e-8, 10.0 * tolerance),
                max_candidates=min(100, len(h_values)),
                max_cuts=budget - added,
                seen_signatures=self._user_cut_signatures,
            )
            for cut in return_cuts:
                callback_model.cbCut(
                    h[cut.corridor]
                    <= cut.big_m * gp.quicksum(y[arc] for arc in cut.crossing_arcs)
                )
                self._user_cut_signatures.add(cut.signature())
                counts["directed_return"] += 1
                added += 1

        for family, count in counts.items():
            self.user_cut_counts[family] = self.user_cut_counts.get(family, 0) + count
        return added

    def _root_user_cut_callback(self, solver: SolverConfig):
        """Create the root-only callback used by the branch-and-cut solve."""

        def callback(callback_model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPNODE:
                return
            if self.root_cut_rounds >= solver.max_root_cut_rounds:
                return
            if callback_model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            if callback_model.cbGet(GRB.Callback.MIPNODE_NODCNT) > 0.5:
                return
            self.root_cut_rounds += 1
            self._add_root_user_cuts(callback_model, solver)

        return callback

    def solve(self, solver: SolverConfig | None = None) -> bool:
        solver = solver or SolverConfig()
        self.solver_config = solver
        self.prepare_algorithm(solver)
        self.network_start_runtime = 0.0
        self.network_start_found = False
        self.network_start_used = False
        self.network_start_key = None
        self.network_start_objective = None
        self.local_branching_runtime = 0.0
        self.local_branching_improved = False
        self.local_branching_objective = None
        self.local_branching_status = None
        self.local_branching_solution_count = 0
        self.local_branching_hamming_distance = None
        self.main_solve_runtime = 0.0
        self.total_runtime = 0.0
        self.user_cut_counts = {}
        self.root_cut_rounds = 0
        self._user_cut_signatures.clear()
        self.model.Params.TimeLimit = solver.time_limit
        self.model.Params.MIPGap = solver.mip_gap
        self.model.Params.Threads = solver.threads
        self.model.Params.OutputFlag = solver.output_flag
        self.model.Params.Seed = solver.seed
        self.model.Params.Method = solver.root_relaxation_method
        start_budget = min(solver.network_start_seconds, 0.20 * solver.time_limit)
        if start_budget > 1e-9 and self.alpha_keys:
            self._construct_network_start(start_budget, solver)
        else:
            self._set_courier_only_start()
        remaining = max(0.1, solver.time_limit - self.network_start_runtime)
        self.model.Params.TimeLimit = remaining
        self.model.Params.MIPGap = solver.mip_gap
        if self.network_start_used:
            self.model.Params.MIPFocus = 1
            self.model.Params.Heuristics = 0.5
            self.model.Params.NoRelHeurTime = min(5.0, 0.05 * remaining)
        else:
            self.model.Params.MIPFocus = 0
            self.model.Params.Heuristics = 0.05
            self.model.Params.NoRelHeurTime = 0.0
        if (
            solver.use_user_cuts
            and solver.max_root_cut_rounds > 0
            and solver.max_cuts_per_round > 0
        ):
            self.model.Params.PreCrush = 1
            self.model.optimize(self._root_user_cut_callback(solver))
        else:
            self.model.optimize()
        self.main_solve_runtime = float(self.model.Runtime)
        self.total_runtime = self.network_start_runtime + self.main_solve_runtime
        return self.model.SolCount > 0

    def _construct_network_start(self, time_limit: float, solver: SolverConfig) -> None:
        """Construct a locker-network MIP start, then restore all bounds.

        The auxiliary neighborhood fixes one display decision and disables
        optional home-assistance offers.  Every bound is released before the
        reported optimization; the paper's objective never changes.
        """

        start_wall_time = time.perf_counter()
        # Pick a high-mass display bundle, using first-order fulfillment saving
        # as the tie-breaker.  The primary volume rule is deliberately robust
        # to costs that are only provisional: it creates the widest feasible
        # service bundle for the exact auxiliary MISOCP to price and repair.
        # No instance labels or previously observed solution are embedded.
        def display_score(key: tuple[str, str]) -> tuple[float, float, tuple[str, str]]:
            destination, locker = key
            saving = 0.0
            volume = 0.0
            for origin, d_node in self.inst.markets:
                if d_node != destination:
                    continue
                amount = (
                    self.inst.h_od[origin, destination]
                    * self.inst.choice_share[origin, destination, locker]
                )
                volume += amount
                unit_saving = max(
                    self.inst.rider_cost(origin, destination)
                    - self.inst.rider_cost(origin, i)
                    - self.inst.config.locker_service_cost
                    - self.inst.config.flight_cost_per_km * self.inst.distance(i, locker)
                    for i in self.inst.locker_origins[origin, destination, locker]
                )
                saving += amount * unit_saving
            return volume, saving, key

        target = max(self.alpha_keys, key=display_score)
        self.network_start_key = target
        alpha_bounds = {
            key: (float(variable.LB), float(variable.UB))
            for key, variable in self.vars["alpha"].items()
        }
        home_upper_bounds = {
            key: float(variable.UB) for key, variable in self.vars["a_hm"].items()
        }
        for key, variable in self.vars["alpha"].items():
            variable.LB = 1.0 if key == target else 0.0
            variable.UB = 1.0 if key == target else 0.0
        for variable in self.vars["a_hm"].values():
            variable.UB = 0.0
        # Ten seconds was enough to cross the linked design/flow feasibility
        # threshold in the benchmark family.  Any additional start budget is
        # spent in the exact-conic Hamming neighbourhood.
        auxiliary_time = min(10.0, time_limit)
        self.model.Params.TimeLimit = auxiliary_time
        self.model.Params.MIPGap = 0.001
        self.model.Params.MIPFocus = 0
        self.model.Params.Heuristics = 0.05
        self.model.Params.NoRelHeurTime = 0.0
        self.model.optimize()
        auxiliary_runtime = float(self.model.Runtime)
        self.network_start_runtime = auxiliary_runtime
        self.network_start_found = self.model.SolCount > 0
        seed = None
        start_values: dict[gp.Var, float] = {}
        if self.network_start_found:
            self.network_start_objective = float(
                sum(expression.getValue() for expression in self.cost_expressions.values())
            )
            start_values = {variable: float(variable.X) for variable in self.model.getVars()}
            # Capture while the auxiliary incumbent is still valid.  Calling
            # model.update() after restoring bounds clears SolCount and X.
            seed = capture_incumbent(self.model, self.vars)
            self.network_start_used = True

        for key, variable in self.vars["alpha"].items():
            variable.LB, variable.UB = alpha_bounds[key]
        for key, variable in self.vars["a_hm"].items():
            variable.UB = home_upper_bounds[key]
        self.model.update()
        local_time = max(0.0, time_limit - (time.perf_counter() - start_wall_time))
        if (
            seed is not None
            and solver.use_local_branching
            and local_time >= 0.1
        ):
            result = improve_mip_start(
                self.model,
                self.vars,
                seed,
                LocalBranchingConfig(
                    radius=solver.local_branching_radius,
                    time_limit=local_time,
                    mip_gap=min(0.01, solver.mip_gap),
                    output_flag=solver.output_flag,
                ),
            )
            self.local_branching_runtime = result.runtime
            self.local_branching_improved = result.improved
            self.local_branching_objective = result.installed_objective
            self.local_branching_status = result.status
            self.local_branching_solution_count = result.solution_count
            self.local_branching_hamming_distance = result.hamming_distance
            self.network_start_objective = result.installed_objective
            self.network_start_runtime = auxiliary_runtime + result.runtime
        elif self.network_start_used:
            self.model.NumStart = 1
            self.model.Params.StartNumber = 0
            for variable, value in start_values.items():
                variable.Start = value
        if self.network_start_used:
            # Retain the network design as start 0 and add the analytically
            # feasible all-courier solution as start 1.  This guarantees that
            # a deliberately exploratory network seed cannot worsen the first
            # incumbent seen by the global search.
            self.model.NumStart = 2
            self.model.update()
            self.model.Params.StartNumber = 1
            self._set_courier_only_start(reset_start_pool=False)
            self.model.Params.StartNumber = 0
        else:
            self._set_courier_only_start()
        self.network_start_runtime = time.perf_counter() - start_wall_time

    def _set_courier_only_start(self, *, reset_start_pool: bool = True) -> None:
        """Seed the always-feasible all-courier counterfactual."""

        if reset_start_pool:
            self.model.NumStart = 1
            self.model.update()
            self.model.Params.StartNumber = 0
        for name, container in self.vars.items():
            if isinstance(container, gp.Var):
                container.Start = 0.0
                continue
            start_value = 1.0 if name in {"queue_v2", "queue_t"} else 0.0
            for variable in container.values():
                variable.Start = start_value

    def write_model(self, path: str) -> None:
        self.build()
        self.model.write(path)
