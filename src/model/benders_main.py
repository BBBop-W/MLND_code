import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import re
import gurobipy as gp
from gurobipy import GRB

# Support both package execution and direct script execution.
if __package__ in (None, ""):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src.data_processor.loader import DataLoader
    from src.utils.output_processor import ResultProcessor
    from src import config
    from src.data_processor.instance import Instance
else:
    from ..data_processor.loader import DataLoader
    from ..utils.output_processor import ResultProcessor
    from .. import config
    from ..data_processor.instance import Instance

if __package__ in (None, ""):
    from src.model.flow_cut_separator import ContinuityFlowSeparator, FlowCut
else:
    from .flow_cut_separator import ContinuityFlowSeparator, FlowCut

XHat = Dict[str, int]
EtaHat = Dict[Tuple[str, int], int]
TOL = 1e-7



class IndexSets:
    """Index sets and helper expressions shared across Benders modules."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.L = instance.L_hat
        self.L_v = instance.L_v if config.USE_VIRTUAL_BASES_OPT else []
        self.L_hat = self.L + self.L_v
        self.P = instance.P
        self.D = instance.D
        self.E = instance.E
        self.N = instance.N
        self.S = list(range(len(config.SERVICE_LEVELS)))
        self.rho = config.SERVICE_LEVELS

        self.A = [arc for arc in instance.A if self._arc_is_allowed(arc)]
        self.A_set = set(self.A)
        self.A_PD = [arc for arc in instance.A_PD if arc in self.A_set]
        self.A_LP = [arc for arc in instance.A_LP if arc in self.A_set]
        self.A_DL = [arc for arc in instance.A_DL if arc in self.A_set]
        self.A_PD_set = set(self.A_PD)
        self.A_LP_set = set(self.A_LP)

        self.out_arcs: Dict[str, list[Tuple[str, str]]] = {}
        self.in_arcs: Dict[str, list[Tuple[str, str]]] = {}
        for arc in self.A:
            self.out_arcs.setdefault(arc[0], []).append(arc)
            self.in_arcs.setdefault(arc[1], []).append(arc)

        self.lp_arcs_from_base = {
            location: [arc for arc in self.A_LP if arc[0] == location]
            for location in self.L_hat
        }
        self.pd_arcs_from_pickup = {
            pickup: [arc for arc in self.A_PD if arc[0] == pickup]
            for pickup in self.P
        }

        self.nearby_bases = {
            pickup: sorted({base for base, node in self.A_LP if node == pickup and base in self.L})
            for pickup in self.P
        }
        self.nearby_bases.update(
            {
                delivery: sorted({base for node, base in self.A_DL if node == delivery and base in self.L})
                for delivery in self.D
            }
        )

        self.nearby_swaps = {
            lv: sorted(
                {
                    swap
                    for _, pickup in self.lp_arcs_from_base.get(lv, [])
                    for swap in instance.E_uv.get((lv, pickup), [])
                }
            )
            for lv in self.L_v
        }
        self.nearby_swaps.update(
            {
                pickup: sorted(
                    {
                        swap
                        for _, delivery in self.pd_arcs_from_pickup.get(pickup, [])
                        for swap in instance.E_uv.get((pickup, delivery), [])
                    }
                )
                for pickup in self.P
            }
        )

        self.H = {
            pickup: sum(
                instance.h.get((pickup, delivery), 0.0)
                for delivery in self.D
                if (pickup, delivery) in self.A_PD_set
            )
            for pickup in self.P
        }

        self.H_hat = {
            delivery: sum(
                instance.h.get((pickup, delivery), 0.0)
                for pickup in self.P
                if (pickup, delivery) in self.A_PD_set
            )
            for delivery in self.D
        }

        self.delivery_demand = {
            delivery: sum(
                instance.h.get((pickup, delivery), 0.0)
                for pickup in self.P
                if (pickup, delivery) in self.A_PD_set
            )
            for delivery in self.D
        }
        self.total_demand = sum(instance.h.values())

        self.phi_keys: list[Tuple[str, str, str]] = []
        self.phi_for_arc: Dict[Tuple[str, str], list[str]] = {}
        self.swap_flow_keys: Dict[str, list[Tuple[str, str, str]]] = {}
        for u_node, v_node in self.A:
            if instance.get_dist(u_node, v_node) <= config.MAX_DIRECT_DISTANCE or (
                u_node in self.D and v_node.startswith("V_")
            ):
                self.phi_keys.append((u_node, v_node, "0"))
                self.phi_for_arc.setdefault((u_node, v_node), []).append("0")
            for swap in instance.E_uv.get((u_node, v_node), []):
                self.phi_keys.append((u_node, v_node, swap))
                self.phi_for_arc.setdefault((u_node, v_node), []).append(swap)
                self.swap_flow_keys.setdefault(swap, []).append((u_node, v_node, swap))

        self.cap_eff = (
            1 - config.ENERGY_SAFETY_MARGIN_EPSILON
        ) * config.BATTERY_CAPACITY_OMEGA_UPPER
        self.energy_exchange_limit = self.cap_eff / config.ENERGY_RATE_OMEGA
        self.service_wait = {
            level: self.rho[level] / (1 - self.rho[level])
            for level in self.S
            if self.rho[level] < 1.0
        }

        self.service_wait_t = {
            level: 1 / (1 - self.rho[level])
            for level in self.S
            if self.rho[level] < 1.0
        }

    @staticmethod
    def _arc_is_allowed(arc: Tuple[str, str]) -> bool:
        if config.USE_VIRTUAL_BASES_OPT:
            return True
        return not arc[0].startswith("V_") and not arc[1].startswith("V_")

    def service_wait_value(
        self,
        eta_hat: EtaHat,
        locations: Optional[Iterable[str]] = None,
    ) -> float:
        active_locations = list(locations) if locations is not None else self.L_hat
        return sum(
            self.service_wait.get(level, 0.0) * eta_hat[(location, level)]
            for location in active_locations
            for level in self.S
        )

    def service_wait_expr(
        self,
        eta_vars: Mapping[Tuple[str, int], gp.Var],
        locations: Optional[Iterable[str]] = None,
    ) -> gp.LinExpr:
        active_locations = list(locations) if locations is not None else self.L_hat
        return gp.quicksum(
            self.service_wait.get(level, 0.0) * eta_vars[location, level]
            for location in active_locations
            for level in self.S
        )

    def outflow_expr(self, psi: Mapping[Tuple[str, str], gp.Var], node: str) -> gp.LinExpr:
        return gp.quicksum(psi[arc] for arc in self.out_arcs.get(node, []))

    def inflow_expr(self, psi: Mapping[Tuple[str, str], gp.Var], node: str) -> gp.LinExpr:
        return gp.quicksum(psi[arc] for arc in self.in_arcs.get(node, []))

    def path_time(self, u_node: str, v_node: str, swap: str) -> float:
        if swap == "0":
            return self.inst.get_time(u_node, v_node)
        return self.inst.get_time(u_node, swap) + self.inst.get_time(swap, v_node)

    def phi_travel_expr(
        self,
        phi: Mapping[Tuple[str, str, str], gp.Var],
        arc_keys: Iterable[Tuple[str, str, str]],
    ) -> gp.LinExpr:
        return gp.quicksum(
            phi[u_node, v_node, swap] * self.path_time(u_node, v_node, swap)
            for u_node, v_node, swap in arc_keys
        )

# ======================================================================
# Full Benders driver
# ======================================================================

class BendersMerge:
    """Benders decomposition with cuts from primal-subproblem duals."""

    def __init__(
        self,
        instance: Instance,
        max_iter: int = 100,
        gap_tol: float = 1e-4,
        time_limit: float = 3600,
        verbose: bool = True,
    ):
        self.inst = instance
        self.data = IndexSets(instance)
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.time_limit = float(time_limit)
        self.verbose = verbose

        self.model: Optional[gp.Model] = None
        self.x: Optional[gp.tupledict] = None
        self.eta: Optional[gp.tupledict] = None
        self.theta: Optional[gp.Var] = None
        self.best_obj = -math.inf
        self.best_sol: Optional[Dict[str, Any]] = None
        self._visited_master_keys: set = set()
        self.iteration_count = 0
        self.callback_calls = 0
        self.callback_total_time = 0.0
        self.solve_wall_time = 0.0
        self.flow_sep_calls = 0
        self.flow_sep_cuts = 0
        self.flow_sep_time = 0.0
        self.core_x: Dict[str, float] = {}
        self.core_eta: Dict[Tuple[str, int], float] = {}
        self.flow_separator = ContinuityFlowSeparator(self.data)
        self._init_core_point()

        # Adaptive MW strategy for INDEPENDENT_MW mode:
        # prefer independent MW cuts; temporarily switch to Pareto MW cuts when stalled.
        self._indep_last_mw_obj: Optional[float] = None
        self._indep_stall_count: int = 0
        self._indep_pareto_rounds_left: int = 0
        self._indep_stall_tol_abs: float = 1e-7
        self._indep_stall_tol_rel: float = 1e-6
        self._indep_stall_trigger_iters: int = 5
        self._indep_pareto_rounds: int = 8

    def _is_mw_obj_stalled(self, mw_obj: float) -> bool:
        if self._indep_last_mw_obj is None:
            return False
        diff = abs(mw_obj - self._indep_last_mw_obj)
        tol = self._indep_stall_tol_abs + self._indep_stall_tol_rel * max(
            1.0, abs(self._indep_last_mw_obj)
        )
        return diff <= tol

    def _init_core_point(self) -> None:
        """Initialize core point: x=1 and eta at rho closest to 0.5."""
        self.core_x = {n: 1.0 for n in self.data.N}
        self.core_x.update({lv: 0.0 for lv in self.data.L_v})
        rho_target = 0.5
        rho_idx = min(
            range(len(config.SERVICE_LEVELS)),
            key=lambda s: abs(config.SERVICE_LEVELS[s] - rho_target),
        )
        self.core_eta = {
            (l, s): (1.0 if s == rho_idx else 0.0)
            for l in self.data.L
            for s in self.data.S
        }
        self.core_eta.update({
            (l, s): 0.0
            for l in self.data.L_v
            for s in self.data.S
            if (l, s) not in self.core_eta}
        )


    def _update_core_point(self, x_hat: Dict[str, int], eta_hat: Dict[Tuple[str, int], int]) -> None:
        """Core-point update: y <- 0.5 y + 0.5 y_hat."""
        for n in self.data.N + self.data.L_v:
            self.core_x[n] = 0.5 * self.core_x.get(n, 1.0) + 0.5 * float(x_hat[n])

        for l in self.data.L_hat:
            for s in self.data.S:
                key = (l, s)
                self.core_eta[key] = 0.5 * self.core_eta.get(key, 0.0) + 0.5 * float(eta_hat[key])

    def _corresponding_e_for_l(self, l_node):

        if not self.data.E:
            return None

        suffix_match = re.search(r"(\d+)$", l_node)
        if suffix_match:
            candidate = f"E{suffix_match.group(1)}"
            if candidate in self.data.E:
                return candidate

        return min(self.data.E, key=lambda e_node: self.data.inst.get_dist(l_node, e_node))


    def _build_master(self) -> None:
        data = self.data
        inst = self.inst

        mdl = gp.Model("BendersMaster_Primal")
        mdl.setParam("OutputFlag", 1 if self.verbose else 0)
        mdl.setParam("TimeLimit", self.time_limit)
        mdl.setParam("Heuristics", 0)
        mdl.setParam("Cuts", 0)

        x = mdl.addVars(data.N + data.L_v, vtype=GRB.BINARY, name="x")
        eta = mdl.addVars(data.L_hat, data.S, vtype=GRB.BINARY, name="eta")
        y = mdl.addVars(data.A_PD, lb=0.0, ub=1.0, name="y")
        x_x = mdl.addVars(data.A_PD, lb=0.0, ub=1.0, name="x_x")

        fixed_cost = gp.quicksum(self.inst.f[u_idx] * x[u_idx] for u_idx in data.N)

        theta = mdl.addVar(lb=-GRB.INFINITY, ub=config.ORDER_REVENUE_COEF * data.total_demand, name="theta")
        mdl.setObjective(theta - fixed_cost, GRB.MAXIMIZE)

        for l_node in data.L:
            e_node = self._corresponding_e_for_l(l_node)
            if e_node is not None:
                mdl.addConstr(x[l_node] <= x[e_node], name=f"link_base_swap_{l_node}_{e_node}")

        for p, d in data.A_PD:
            mdl.addConstr(x_x[p, d] <= x[p], name=f"xx_up_p_{p}_{d}")
            mdl.addConstr(x_x[p, d] <= x[d], name=f"xx_up_d_{p}_{d}")
            mdl.addConstr(x_x[p, d] >= x[p] + x[d] - 1, name=f"xx_low_{p}_{d}")


        # service level selection
        for l in data.L_hat:
            mdl.addConstr(
                gp.quicksum(eta[l, s] for s in data.S) == x[l],
                name=f"svc_{l}",
            )

        # connectivity
        for n in data.P + data.D:
            bases = data.nearby_bases.get(n, [])
            if bases:
                mdl.addConstr(
                    x[n] <= gp.quicksum(x[b] for b in bases),
                    name=f"near_base_{n}",
                )
            else:
                mdl.addConstr(x[n] == 0, name=f"forbid_{n}")

        for n in data.L_v + data.P:
            swaps = data.nearby_swaps.get(n, [])
            if swaps:
                mdl.addConstr(
                    x[n] <= gp.quicksum(x[e] for e in swaps),
                    name=f"near_swap_{n}",
                )
            else:
                mdl.addConstr(x[n] == 0, name=f"forbid_swap_{n}")

        if config.USE_VIRTUAL_BASES_OPT:
            for d in data.D:
                mdl.addConstr(
                    x[inst.d_to_lv[d]] <= x[d],
                    name=f"vbase_{d}",
                )

        mw = data.service_wait_expr(eta)

        mdl.addConstr(
            theta <= config.ORDER_REVENUE_COEF * gp.quicksum(data.H[i] * x[i] for i in data.P),
            name="theta_P",
        )
        mdl.addConstr(
            theta <= config.ORDER_REVENUE_COEF * gp.quicksum(data.H_hat[j] * x[j] for j in data.D),
            name="theta_D",
        )
        mdl.addConstr(
            theta <= config.ORDER_REVENUE_COEF * gp.quicksum(inst.h[i, j] * y[i, j] for i, j in data.A_PD),
            name="theta_PD",
        )

        mdl._vars_x = x
        mdl._vars_eta = eta
        mdl._vars_y = y
        mdl._vars_xx = x_x
        self.model = mdl
        self.x = x
        self.eta = eta
        self.theta = theta
        self.x_x = x_x

    def get_solution_binary(self, model: gp.Model):
        x_hat = {
            n: 1 if model.cbGetSolution(self.x[n]) > 0.5 else 0
            for n in self.data.N + self.data.L_v
        }
        eta_hat = {
            (l, s): 1 if model.cbGetSolution(self.eta[l, s]) > 0.5 else 0
            for l in self.data.L_hat
            for s in self.data.S
        }
        y_hat = {
            (i, j): float(model.cbGetSolution(model._vars_y[i, j]))
            for i, j in self.data.A_PD
        }
        tv = model.cbGetSolution(self.theta)
        return x_hat, eta_hat, y_hat, tv

    def _master_solution_key(self, x_hat, eta_hat):
        return (
            tuple(x_hat[n] for n in self.data.N + self.data.L_v),
            tuple(eta_hat[l, s] for l in self.data.L_hat for s in self.data.S),
        )

    def _build_cut_rhs(self, duals: dict, x_ref, eta_ref, q_ref, use_gurobi_expr: bool):
        """Build cut RHS with the same objective builder used in DualSubproblem."""
        if __package__ in (None, ""):
            from src.model.benders_dual import DualSubproblem
        else:
            from .benders_dual import DualSubproblem
        return DualSubproblem.build_objective_expr(
            data=self.data,
            inst=self.inst,
            x_ref=x_ref,
            eta_ref=eta_ref,
            q_ref=q_ref,
            alpha1=duals["alpha1"],
            alpha2=duals["alpha2"],
            alpha4=duals["alpha4"],
            alpha5=duals["alpha5"],
            epsilon2=duals["epsilon2"],
            epsilon3=duals["epsilon3"],
            beta2=duals["beta2"],
            pi1=duals["pi1"],
            pi2=duals["pi2"],
            pi3=duals["pi3"],
            delta1=duals["delta1"],
            delta2=duals["delta2"],
            Delta=duals["Delta"],
            sigma1=duals["sigma1"],
            kappa1=duals["kappa1"],
            kappa3=duals["kappa3"],
            use_gurobi_expr=use_gurobi_expr,
        )

    def _cut_check(self, cd: dict) -> float:
        """Evaluate affine cut RHS at (x_hat, eta_hat) for consistency check."""
        duals = cd["duals"]
        data = self.data
        x_hat = cd["x_hat"]
        eta_hat = cd["eta_hat"]
        q_hat = {loc: data.service_wait_value(eta_hat, [loc]) for loc in data.L_hat}
        return self._build_cut_rhs(duals, x_hat, eta_hat, q_hat, use_gurobi_expr=False)

    def _add_cut(self, model: gp.Model, cut_data: dict) -> None:
        """Add optimality cut using coefficients computed from subproblem dual multipliers."""
        duals = cut_data["duals"]
        data = self.data
        Q_var = {loc: data.service_wait_value(self.eta, [loc]) for loc in data.L_hat}
        expr = self._build_cut_rhs(duals, self.x, self.eta, Q_var, use_gurobi_expr=True)

        model.cbLazy(self.theta <= expr)



    def _add_nogood(self, model: gp.Model, cd: dict) -> None:
        """Feasibility cut: exclude the current binary combination."""
        xh = cd["x_hat"]
        ones = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 1]
        zeros = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 0]
        model.cbLazy(
            gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1
        )

    def _add_flow_cut(self, model: gp.Model, flow_cut: FlowCut) -> None:
        lhs = gp.quicksum(
            self.flow_separator.demand_coeff(lv, s) * self.eta[lv, s]
            for lv in flow_cut.lhs_lvs
            for s in self.data.S
        )
        rhs = flow_cut.rhs_const + gp.quicksum(
            coef * self.x[node] for node, coef in flow_cut.rhs_x_coeff.items()
        )
        rhs += gp.quicksum(
            coef * self.x_x[p, d] for (p, d), coef in flow_cut.rhs_xx_coeff.items()
        )
        model.cbLazy(lhs <= rhs)

    def solve(self) -> Optional[float]:
        self._build_master()
        if __package__ in (None, ""):
            from src.model.benders_primal import PrimalSubproblem
            from src.model.benders_dual import DualSubproblem
        else:
            from .benders_primal import PrimalSubproblem
            from .benders_dual import DualSubproblem
        subPrimal = PrimalSubproblem(self.inst, self.data)
        subDual = DualSubproblem(self.inst, self.data)

        def callback(model: gp.Model, where: int):
            if where != GRB.Callback.MIPSOL:
                return
            cb_t0 = time.perf_counter()
            self.callback_calls += 1
            try:
                xh, etah, yh, tv = self.get_solution_binary(model)
                master_key = self._master_solution_key(xh, etah)
                repeated_master = master_key in self._visited_master_keys

                sep_t0 = time.perf_counter()
                self.flow_sep_calls += 1
                flow_cut = self.flow_separator.separate(xh, etah)
                self.flow_sep_time += time.perf_counter() - sep_t0
                if flow_cut is not None:
                    self.flow_sep_cuts += 1
                    if self.verbose:
                        print(
                            "[FlowCut] "
                            f"required={flow_cut.required:.6f}, cut_value={flow_cut.cut_value:.6f}, "
                            f"lhs_lvs={len(flow_cut.lhs_lvs)}"
                        )
                    self._add_flow_cut(model, flow_cut)
                    return

                if not repeated_master:
                    self._visited_master_keys.add(master_key)
                elif not config.INDEPENDENT_MW:
                    return
                self.iteration_count += 1

                # status, obj, cd = subPrimal.solve(xh, etah)
                status, obj, cd = subDual.solve(xh, etah)
                incumbent_cd = cd

                mw_status = None
                mw_obj = None
                mw_cd = None
                use_incumbent_fallback_cut = False
                mw_mode = "none"

                if obj is not None:
                    if config.INDEPENDENT_MW:
                        if self._indep_pareto_rounds_left > 0:
                            mw_mode = "pareto"
                            mw_status, mw_obj, mw_cd = subDual.solve_mw(
                                x_hat=xh,
                                eta_hat=etah,
                                core_x=self.core_x,
                                core_eta=self.core_eta,
                                z_star=obj,
                            )
                            self._indep_pareto_rounds_left -= 1
                            if self._indep_pareto_rounds_left == 0 and self.verbose:
                                print(
                                    "[MW-Switch] Pareto window ended; switching back to independent MW."
                                )
                        else:
                            mw_mode = "independent"
                            mw_status, mw_obj, mw_cd = subDual.solve_mw_independent(
                                core_x=self.core_x,
                                core_eta=self.core_eta,
                            )

                            if mw_status == GRB.OPTIMAL and mw_obj is not None:
                                if self._is_mw_obj_stalled(mw_obj):
                                    self._indep_stall_count += 1
                                else:
                                    self._indep_stall_count = 0
                                self._indep_last_mw_obj = mw_obj

                                if self._indep_stall_count >= self._indep_stall_trigger_iters:
                                    self._indep_pareto_rounds_left = self._indep_pareto_rounds
                                    self._indep_stall_count = 0
                                    if self.verbose:
                                        print(
                                            "[MW-Switch] independent MW stalled; "
                                            f"switching to Pareto MW for {self._indep_pareto_rounds} iterations."
                                        )
                            else:
                                self._indep_stall_count = 0
                    else:
                        mw_status, mw_obj, mw_cd = subDual.solve_mw(
                            x_hat=xh,
                            eta_hat=etah,
                            core_x=self.core_x,
                            core_eta=self.core_eta,
                            z_star=obj,
                        )
                        if mw_status == GRB.OPTIMAL and mw_cd is not None:
                            cd = mw_cd
                        else:
                            use_incumbent_fallback_cut = True

                if self.verbose:
                    mw_obj_str = f"{mw_obj:.6f}" if mw_obj is not None else "None"
                    print(
                        "[Benders] "
                        f"iter={self.iteration_count}, theta={tv:.6f}, "
                        f"sub_status={status}, sub_obj={(f'{obj:.6f}' if obj is not None else 'None')}, "
                        f"mw_status={mw_status}, mw_obj={mw_obj_str}, mw_mode={mw_mode}"
                    )

            # if obj is not None:
            #     rhs_at_inc = self._cut_check(cd)
            #     tol = 1e-5 * max(1.0, abs(obj))
            #     if abs(rhs_at_inc - obj) > tol:
            #         status_primal, obj_primal, cd_primal = subPrimal.solve(xh, etah)
            #         print(
            #                 "Inconsistent cut: "
            #                 f"RHS(x_hat,eta_hat)={rhs_at_inc:.6f}, dual_obj={obj:.6f}, primal_obj={obj_primal:.6f}"
            #             )
                # else:
                #     print(
                #             f"Cut check passed: RHS(x_hat,eta_hat)={rhs_at_inc:.6f}, dual_obj={obj:.6f}"
                #         )
                    
                if obj is None:
                    added = False

                    if not added:
                        if self.verbose:
                            print("Infeasible subproblem with no structural certificate, adding nogood cut.")
                        _status, _obj, _cd = subPrimal.solve(xh, etah)
                        self._add_nogood(model, cd)
                    return

                # Convergence check at incumbent for max-theta form: no cut needed if theta already below v(y*).
                if obj is not None and tv <= obj + config.EPS:
                    return

                if cd["kind"] == "feasibility":
                    self._add_nogood(model, cd)
                    return
                
                if obj is not None:
                    if config.INDEPENDENT_MW:
                        # Always add incumbent cut so the current violating incumbent is cut.
                        self._add_cut(model, cd)
                        if mw_status == GRB.OPTIMAL and mw_cd is not None:
                            # Add independent core-point (MW) cut as a strengthening cut.
                            self._add_cut(model, mw_cd)
                        elif self.verbose:
                            print(
                                f"INDEPENDENT_MW: MW status={mw_status}; using incumbent cut only."
                            )
                    else:
                        if use_incumbent_fallback_cut and self.verbose:
                            print(
                                f"MW status={mw_status}; fallback to incumbent subproblem cut."
                            )
                        self._add_cut(model, incumbent_cd if use_incumbent_fallback_cut else cd)

                if obj is not None and obj > self.best_obj:
                    self.best_obj = obj
                    self.best_sol = {"x": xh, "eta": etah, "obj": obj}

                if obj is not None and self.best_sol is not None:
                    self._update_core_point(self.best_sol["x"], self.best_sol["eta"])
            finally:
                self.callback_total_time += time.perf_counter() - cb_t0

        self.model.setParam("LazyConstraints", 1)
        solve_t0 = time.perf_counter()
        self.model.optimize(callback)
        self.solve_wall_time = time.perf_counter() - solve_t0
        self.model.write("benders_primal_master.lp")

        if self.verbose:
            stats = subDual.get_stats()
            print("[Benders] Subproblem statistics:")
            print(
                f"  sub_calls={int(stats['sub_calls'])}, sub_optimal={int(stats['sub_optimal'])}, "
                f"sub_nonoptimal={int(stats['sub_nonoptimal'])}, sub_avg_time={stats['sub_avg_time']:.4f}s"
            )
            print(
                f"  mw_calls={int(stats['mw_calls'])}, mw_optimal={int(stats['mw_optimal'])}, "
                f"mw_nonoptimal={int(stats['mw_nonoptimal'])}, mw_avg_time={stats['mw_avg_time']:.4f}s"
            )

            total_time = max(self.solve_wall_time, 1e-12)
            sub_time = stats["sub_total_time"]
            mw_time = stats["mw_total_time"]
            cb_time = self.callback_total_time
            cb_other_time = max(0.0, cb_time - sub_time - mw_time)
            master_time = max(0.0, total_time - cb_time)

            print("[Benders] Timing analysis:")
            print(
                f"  iterations={self.iteration_count}, callback_calls={self.callback_calls}, "
                f"time_limit={self.time_limit:.1f}s"
            )
            print(f"  total_wall_time={total_time:.4f}s")
            print(
                f"  master_time={master_time:.4f}s "
                f"({100.0 * master_time / total_time:.2f}%)"
            )
            print(
                f"  subproblem_time={sub_time:.4f}s "
                f"({100.0 * sub_time / total_time:.2f}%)"
            )
            print(
                f"  mw_time={mw_time:.4f}s "
                f"({100.0 * mw_time / total_time:.2f}%)"
            )
            print(
                f"  callback_other_time={cb_other_time:.4f}s "
                f"({100.0 * cb_other_time / total_time:.2f}%)"
            )
            print(
                f"  flow_sep_calls={self.flow_sep_calls}, flow_sep_cuts={self.flow_sep_cuts}, "
                f"flow_sep_time={self.flow_sep_time:.4f}s "
                f"({100.0 * self.flow_sep_time / total_time:.2f}%)"
            )

        return self.model.ObjVal if self.model.Status == GRB.OPTIMAL else None




if __name__ == "__main__":
    print("=" * 70)
    print("LOADING INSTANCE")
    print("=" * 70)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    locations_path = os.path.join(base_dir, "data", "instances", "Meituan", "locations.csv")
    orders_path = os.path.join(base_dir, "data", "instances", "Meituan", "orders.csv")
    inst = DataLoader().load_instance(
        locations_path,
        orders_path,
    )
    print(f"Nodes: P={len(inst.P)}, D={len(inst.D)}, L={len(inst.L_hat)}")
    print(f"Total Demand: {sum(inst.h.values()):.2f}")

    print("\n" + "=" * 70)
    print("BENDERS test")
    print("=" * 70)
    try:
        primal_solver = BendersMerge(inst, max_iter=50, verbose=True, time_limit=7200)
        primal_obj = primal_solver.solve()
        print(f"\nPrimal Benders objective: {primal_obj}")
        if primal_solver.best_sol:
            print(f"Best solution from primal approach: {primal_solver.best_obj}")
    except Exception as e:
        print(f"ERROR in primal approach: {e}")
        import traceback
        traceback.print_exc()

    processor_mei = ResultProcessor(primal_solver, inst)
    processor_mei.print_summary()