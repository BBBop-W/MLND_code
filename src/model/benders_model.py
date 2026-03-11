"""Decomposition algorithms for the MLND model.

This module now distinguishes between:

* ``QueueOuterApproximation``: the previous OA algorithm that iteratively
  tightens tangent cuts for the convex queueing term.
* ``BendersDecomposition``: a binary-master / continuous-subproblem Benders
  decomposition. The subproblem is solved in primal form as a convex QCP,
  and master cuts are generated from subproblem duals.
"""

import math
import time

import gurobipy as gp
from gurobipy import GRB

from .. import config
from ..data_processor.instance import Instance


class _ModelData:
    """Precomputed index sets shared by decomposition algorithms."""

    def __init__(self, instance: Instance):
        self.inst = instance

        self.L_hat = instance.L_hat
        self.L_v = instance.L_v if config.USE_VIRTUAL_BASES_OPT else []
        self.L = self.L_hat + self.L_v
        self.P = instance.P
        self.D = instance.D
        self.E = instance.E
        self.N = instance.N
        self.S = list(range(len(config.SERVICE_LEVELS)))
        self.rho = config.SERVICE_LEVELS

        allowed_arc = (
            (lambda arc: True)
            if config.USE_VIRTUAL_BASES_OPT
            else (lambda arc: not arc[0].startswith("V_") and not arc[1].startswith("V_"))
        )

        self.A_all = [arc for arc in instance.A if allowed_arc(arc)]
        self.A_all_set = set(self.A_all)
        self.A_PD = [arc for arc in instance.A_PD if arc in self.A_all_set]
        self.A_LP = [arc for arc in instance.A_LP if arc in self.A_all_set]
        self.A_DL = [arc for arc in instance.A_DL if arc in self.A_all_set]
        self.A = self.A_PD + self.A_LP + self.A_DL
        self.A_set = set(self.A)

        self.E_uv = {arc: instance.E_uv[arc] for arc in self.A if arc in instance.E_uv}
        self.D_set = set(self.D)
        self.L_hat_set = set(self.L_hat)
        self.A_PD_set = set(self.A_PD)
        self.A_LP_set = set(self.A_LP)

        self.out_arcs = {}
        self.in_arcs = {}
        for u_node, v_node in self.A:
            self.out_arcs.setdefault(u_node, []).append((u_node, v_node))
            self.in_arcs.setdefault(v_node, []).append((u_node, v_node))

        self.nearby_bases = {}
        for p_node in self.P:
            self.nearby_bases[p_node] = sorted({l_node for l_node, i_node in self.A_LP if i_node == p_node})
        for d_node in self.D:
            self.nearby_bases[d_node] = sorted({l_node for u_node, l_node in self.A_DL if u_node == d_node})

        self.nearby_swaps = {}
        for lv_node in self.L_v:
            self.nearby_swaps[lv_node] = sorted(
                {
                    k_node
                    for u_node, i_node in self.A_LP
                    if u_node == lv_node
                    for k_node in self.E_uv.get((u_node, i_node), [])
                }
            )
        for p_node in self.P:
            self.nearby_swaps[p_node] = sorted(
                {
                    k_node
                    for u_node, d_node in self.A_PD
                    if u_node == p_node
                    for k_node in self.E_uv.get((u_node, d_node), [])
                }
            )

        self.H = {}
        for i_node in self.P:
            self.H[i_node] = sum(
                instance.h.get((i_node, j_node), 0.0)
                for j_node in self.D
                if (i_node, j_node) in self.A_PD_set
            )
        self.delivery_demand = {}
        for j_node in self.D:
            self.delivery_demand[j_node] = sum(
                instance.h.get((i_node, j_node), 0.0)
                for i_node in self.P
                if (i_node, j_node) in self.A_PD_set
            )

        self.phi_keys = []
        for u_node, v_node in self.A:
            dist_uv = instance.get_dist(u_node, v_node)
            if dist_uv <= config.MAX_DIRECT_DISTANCE or (
                u_node in self.D_set and v_node.startswith("V_")
            ):
                self.phi_keys.append((u_node, v_node, "0"))
            for k_node in self.E_uv.get((u_node, v_node), []):
                self.phi_keys.append((u_node, v_node, k_node))
        self.phi_keys_set = set(self.phi_keys)

        self.phi_for_arc = {}
        self.phi_through_swap = {}
        for u_node, v_node, k_node in self.phi_keys:
            self.phi_for_arc.setdefault((u_node, v_node), []).append(k_node)
            if k_node != "0":
                self.phi_through_swap.setdefault(k_node, []).append((u_node, v_node, k_node))

        self.total_demand = sum(instance.h.get((i_node, j_node), 0.0) for i_node, j_node in self.A_PD)
        self.cap_eff = (
            (1 - config.ENERGY_SAFETY_MARGIN_EPSILON)
            * config.BATTERY_CAPACITY_OMEGA_UPPER
        )
        self.service_wait = {
            s_idx: self.rho[s_idx] / (1 - self.rho[s_idx])
            for s_idx in self.S
            if self.rho[s_idx] < 1.0
        }


class QueueOuterApproximation:
    """OA algorithm for the queueing term in the full MILP master."""

    def __init__(
        self,
        instance: Instance,
        max_iter=50,
        gap_tol=1e-4,
        time_limit=3600,
        mip_tl=300,
        verbose=True,
    ):
        self.inst = instance
        self.data = _ModelData(instance)
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.time_limit = time_limit
        self.mip_tl = mip_tl
        self.verbose = verbose

        self.best_obj = None
        self.history = []

    def _build_master(self):
        data = self.data
        inst = self.inst

        m = gp.Model("Queue_Outer_Approximation")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", self.mip_tl)

        x_nodes = data.N + data.L_v
        x = m.addVars(x_nodes, vtype=GRB.BINARY, name="x")
        u = m.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
        y = m.addVars(data.A_LP, lb=0.0, ub=1.0, name="y")
        w_keys = [(l_node, i_node, j_node) for l_node, i_node in data.A_LP for j_node in data.D if (i_node, j_node) in data.A_PD_set]
        w = m.addVars(w_keys, lb=0.0, name="w")
        D_var = m.addVars(data.A_LP, lb=0.0, name="D")
        z_keys = [(l_node, i_node, s_idx) for l_node, i_node in data.A_LP for s_idx in data.S]
        z = m.addVars(z_keys, lb=0.0, name="z")
        psi = m.addVars(data.A, lb=0.0, name="psi")
        phi = m.addVars(data.phi_keys, lb=0.0, name="phi")
        xi = m.addVars(data.P + data.D, lb=0.0, name="xi")
        lam = m.addVars(data.P + data.D, lb=0.0, name="lam")
        gamma = m.addVar(lb=0.0, name="gamma")

        m.setObjective(gp.quicksum(psi[i_node, j_node] for i_node, j_node in data.A_PD), GRB.MAXIMIZE)

        for i_node in data.P:
            ls = [l_node for l_node in data.L if (l_node, i_node) in data.A_LP_set]
            if ls:
                m.addConstr(gp.quicksum(y[l_node, i_node] for l_node in ls) <= x[i_node])
        for l_node, i_node in data.A_LP:
            m.addConstr(y[l_node, i_node] <= x[l_node])
        for l_node in data.L:
            m.addConstr(gp.quicksum(u[l_node, s_idx] for s_idx in data.S) == x[l_node])

        for node in data.P + data.D:
            nearby_bases = data.nearby_bases.get(node, [])
            if nearby_bases:
                m.addConstr(x[node] <= gp.quicksum(x[l_node] for l_node in nearby_bases), name=f"nearby_base_{node}")
            else:
                m.addConstr(x[node] == 0, name=f"nearby_base_forbid_{node}")

        for node in data.L_v + data.P:
            nearby_swaps = data.nearby_swaps.get(node, [])
            if nearby_swaps:
                m.addConstr(x[node] <= gp.quicksum(x[k_node] for k_node in nearby_swaps), name=f"nearby_swap_{node}")
            else:
                m.addConstr(x[node] == 0, name=f"nearby_swap_forbid_{node}")

        if config.USE_VIRTUAL_BASES_OPT:
            for d_node in data.D:
                m.addConstr(x[inst.d_to_lv[d_node]] <= x[d_node])

        for l_node, i_node in data.A_LP:
            H_i = data.H.get(i_node, 0.0)
            for j_node in data.D:
                if (i_node, j_node) not in data.A_PD_set:
                    continue
                m.addConstr(w[l_node, i_node, j_node] <= y[l_node, i_node])
                m.addConstr(w[l_node, i_node, j_node] <= x[j_node])
                m.addConstr(w[l_node, i_node, j_node] >= y[l_node, i_node] - (1 - x[j_node]))
            m.addConstr(
                D_var[l_node, i_node]
                == gp.quicksum(
                    inst.h.get((i_node, j_node), 0.0) * w[l_node, i_node, j_node]
                    for j_node in data.D
                    if (i_node, j_node) in data.A_PD_set
                )
            )
            for s_idx in data.S:
                m.addConstr(z[l_node, i_node, s_idx] <= D_var[l_node, i_node])
                m.addConstr(z[l_node, i_node, s_idx] <= H_i * u[l_node, s_idx])
                m.addConstr(z[l_node, i_node, s_idx] >= D_var[l_node, i_node] - H_i * (1 - u[l_node, s_idx]))
            m.addConstr(
                psi[l_node, i_node] == gp.quicksum(data.rho[s_idx] * z[l_node, i_node, s_idx] for s_idx in data.S)
            )

        for v_node in data.N + data.L_v:
            m.addConstr(
                gp.quicksum(psi[arc] for arc in data.in_arcs.get(v_node, []))
                == gp.quicksum(psi[arc] for arc in data.out_arcs.get(v_node, []))
            )

        for i_node, j_node in data.A_PD:
            m.addConstr(psi[i_node, j_node] <= inst.h[i_node, j_node])

        for u_node, v_node in data.A:
            m.addConstr(
                psi[u_node, v_node]
                == gp.quicksum(phi[u_node, v_node, k_node] for k_node in data.phi_for_arc.get((u_node, v_node), []))
            )

        for nd in data.P + data.D:
            m.addConstr(
                gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, []))
                <= config.MAX_WAITING_FACTOR_P * x[nd]
            )

        for k_node in data.E:
            m.addConstr(
                gp.quicksum(phi[key] for key in data.phi_through_swap.get(k_node, []))
                <= config.MAX_FLOW_E * x[k_node]
            )

        if config.USE_VIRTUAL_BASES_OPT:
            for lv_node in data.L_v:
                rr_expr = gp.quicksum(
                    data.service_wait[s_idx] * u[lv_node, s_idx]
                    for s_idx in data.service_wait
                )
                lp_arcs = [(u_node, i_node) for u_node, i_node in data.A_LP if u_node == lv_node]
                m.addConstr(rr_expr <= config.MAX_WAITING_FACTOR_LV * gp.quicksum(psi[arc] for arc in lp_arcs))

        oa_pts = {}
        for nd in data.P + data.D:
            m.addConstr(lam[nd] == gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, [])))
            m.addConstr(lam[nd] <= inst.mu[nd] - 1e-3)
            oa_pts[nd] = []
            for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                lam0 = inst.mu[nd] * frac
                oa_pts[nd].append(lam0)
                den = inst.mu[nd] - lam0
                f0 = lam0 / den
                fp = inst.mu[nd] / (den * den)
                m.addConstr(xi[nd] >= (f0 - fp * lam0) + fp * lam[nd])

        if config.USE_VIRTUAL_BASES_OPT:
            for lv_node in data.L_v:
                wait_term = gp.quicksum(
                    data.service_wait[s_idx] * u[lv_node, s_idx]
                    for s_idx in data.service_wait
                )
                transit = gp.LinExpr()
                replen = gp.LinExpr()
                for u_node, i_node in data.A_LP:
                    if u_node != lv_node:
                        continue
                    if (u_node, i_node, "0") in data.phi_keys_set:
                        transit += phi[u_node, i_node, "0"] * inst.get_time(u_node, i_node)
                    for k_node in data.E_uv.get((u_node, i_node), []):
                        if (u_node, i_node, k_node) in data.phi_keys_set:
                            travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, i_node)
                            transit += phi[u_node, i_node, k_node] * travel_time
                            replen += phi[u_node, i_node, k_node]
                m.addConstr(config.ENERGY_RATE_OMEGA * (wait_term + transit) <= data.cap_eff * replen)

        for i_node in data.P:
            transit = gp.LinExpr()
            replen = gp.LinExpr()
            for u_node, v_node in data.A_PD:
                if u_node != i_node:
                    continue
                if (u_node, v_node, "0") in data.phi_keys_set:
                    transit += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
                for k_node in data.E_uv.get((u_node, v_node), []):
                    if (u_node, v_node, k_node) in data.phi_keys_set:
                        travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                        transit += phi[u_node, v_node, k_node] * travel_time
                        replen += phi[u_node, v_node, k_node]
            m.addConstr(config.ENERGY_RATE_OMEGA * (xi[i_node] + transit) <= data.cap_eff * replen)

        for j_node in data.D:
            transit = gp.LinExpr()
            replen = gp.LinExpr()
            for u_node, v_node in data.out_arcs.get(j_node, []):
                if (u_node, v_node, "0") in data.phi_keys_set:
                    transit += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
                    if v_node in data.L_hat_set:
                        replen += phi[u_node, v_node, "0"]
                for k_node in data.E_uv.get((u_node, v_node), []):
                    if (u_node, v_node, k_node) in data.phi_keys_set:
                        travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                        transit += phi[u_node, v_node, k_node] * travel_time
                        replen += phi[u_node, v_node, k_node]
            m.addConstr(config.ENERGY_RATE_OMEGA * (xi[j_node] + transit) <= data.cap_eff * replen)

        fleet_expr = gp.LinExpr()
        fleet_expr += gp.quicksum(
            data.service_wait[s_idx] * u[l_node, s_idx]
            for l_node in data.L
            for s_idx in data.service_wait
        )
        fleet_expr += gp.quicksum(xi[nd] for nd in data.P + data.D)
        for u_node, v_node in data.A:
            if (u_node, v_node, "0") in data.phi_keys_set:
                fleet_expr += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
            for k_node in data.E_uv.get((u_node, v_node), []):
                if (u_node, v_node, k_node) in data.phi_keys_set:
                    travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                    fleet_expr += phi[u_node, v_node, k_node] * travel_time
        m.addConstr(fleet_expr <= gamma)
        m.addConstr(gp.quicksum(inst.f[n_node] * x[n_node] for n_node in data.N) + config.COST_DRONE * gamma <= config.BUDGET)
        m.addConstr(gamma <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[l_node] for l_node in data.L_hat))

        m.update()

        self._m = m
        self._xi = xi
        self._lam = lam
        self._oa_pts = oa_pts
        self._psi = psi

    def _add_oa_cuts(self):
        max_viol = 0.0
        cuts_added = 0
        for nd in self.data.P + self.data.D:
            lam_val = self._lam[nd].X
            xi_val = self._xi[nd].X
            mu_i = self.inst.mu[nd]

            if lam_val < 1e-8 or lam_val >= mu_i - 1e-6:
                continue

            xi_exact = lam_val / (mu_i - lam_val)
            viol = xi_exact - xi_val
            if viol <= 1e-6:
                continue

            max_viol = max(max_viol, viol)
            if any(abs(lam_val - existing) < 0.5 for existing in self._oa_pts[nd]):
                continue

            self._oa_pts[nd].append(lam_val)
            den = mu_i - lam_val
            f0 = lam_val / den
            fp = mu_i / (den * den)
            self._m.addConstr(self._xi[nd] >= (f0 - fp * lam_val) + fp * self._lam[nd])
            cuts_added += 1

        return max_viol, cuts_added

    def solve(self):
        start = time.time()
        if self.verbose:
            print("=" * 72)
            print("QUEUE OUTER APPROXIMATION FOR MLND MODEL")
            print("=" * 72)
            print("Building OA master...")

        self._build_master()

        if self.verbose:
            print(f"Master: {self._m.NumVars} vars, {self._m.NumConstrs} constrs")

        prev_obj = None
        for iteration in range(1, self.max_iter + 1):
            elapsed = time.time() - start
            if elapsed > self.time_limit:
                break

            self._m.setParam("TimeLimit", min(self.mip_tl, self.time_limit - elapsed))
            self._m.optimize()

            if self._m.SolCount == 0:
                break

            obj_val = self._m.ObjVal
            mip_gap = self._m.MIPGap if hasattr(self._m, "MIPGap") else 0.0
            max_viol, cuts_added = self._add_oa_cuts()

            self.best_obj = obj_val
            self.history.append(
                {
                    "iter": iteration,
                    "obj": obj_val,
                    "gap": mip_gap,
                    "max_viol": max_viol,
                    "cuts": cuts_added,
                    "time": elapsed,
                }
            )

            if self.verbose:
                print(
                    f"Iter {iteration:3d} | Obj={obj_val:10.4f} | MIPGap={mip_gap:8.4%} | "
                    f"MaxViol={max_viol:10.6f} | NewCuts={cuts_added:3d} | {elapsed:.1f}s"
                )

            if cuts_added == 0 and max_viol <= self.gap_tol:
                break
            if prev_obj is not None and abs(obj_val - prev_obj) < 1e-6 and cuts_added == 0:
                break
            prev_obj = obj_val

        if self.verbose:
            print("=" * 72)
            print(f"Done  best={self.best_obj:.4f}  time={time.time() - start:.1f}s")
            print("=" * 72)
        return self.best_obj

    def get_solution(self):
        if self._m.SolCount == 0:
            return None

        sol = {"objective": self.best_obj, "history": self.history, "x": {}, "u": {}}
        for var in self._m.getVars():
            if var.VarName.startswith("x["):
                sol["x"][var.VarName[2:-1]] = var.X
            elif var.VarName.startswith("u["):
                parts = var.VarName[2:-1].split(",")
                sol["u"][(parts[0], int(parts[1]))] = var.X

        gamma_var = self._m.getVarByName("gamma")
        sol["gamma"] = gamma_var.X if gamma_var is not None else 0.0
        return sol


class BendersDecomposition:
    """Callback-based binary-master / convex-subproblem Benders decomposition.

    The master contains only binary design variables ``x`` and ``u`` plus the
    served-demand estimate ``theta``. Candidate incumbents are separated inside
    a Gurobi callback: the continuous convex QCP subproblem is solved in primal
    form and then contributes either a lazy Benders optimality cut or, if the
    fixed design is infeasible, a lazy no-good cut.
    """

    def __init__(
        self,
        instance: Instance,
        max_iter=100,
        gap_tol=1e-4,
        time_limit=3600,
        master_time_limit=300,
        verbose=True,
    ):
        self.inst = instance
        self.data = _ModelData(instance)
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.time_limit = time_limit
        self.master_time_limit = master_time_limit
        self.verbose = verbose

        self.best_obj = None
        self.best_master_obj = None
        self.best_solution = None
        self.history = []
        self.callback_stats = {"incumbents": 0, "optimality": 0, "incumbent": 0, "nogood": 0}
        self._evaluated_designs = set()

    def _build_master(self):
        data = self.data
        m = gp.Model("MLND_Benders_Master")
        m.setParam("OutputFlag", 0)

        x_nodes = data.N + data.L_v
        x = m.addVars(x_nodes, vtype=GRB.BINARY, name="x")
        u = m.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
        theta = m.addVar(lb=0.0, ub=data.total_demand, name="theta")

        m.setObjective(theta, GRB.MAXIMIZE)

        for l_node in data.L:
            m.addConstr(gp.quicksum(u[l_node, s_idx] for s_idx in data.S) == x[l_node], name=f"service_select_{l_node}")

        for node in data.P + data.D:
            nearby_bases = data.nearby_bases.get(node, [])
            if nearby_bases:
                m.addConstr(
                    x[node] <= gp.quicksum(x[l_node] for l_node in nearby_bases),
                    name=f"hier_base_{node}",
                )
            else:
                m.addConstr(x[node] == 0, name=f"hier_base_forbid_{node}")

        for node in data.L_v + data.P:
            nearby_swaps = data.nearby_swaps.get(node, [])
            if nearby_swaps:
                m.addConstr(
                    x[node] <= gp.quicksum(x[k_node] for k_node in nearby_swaps),
                    name=f"hier_swap_{node}",
                )
            else:
                m.addConstr(x[node] == 0, name=f"hier_swap_forbid_{node}")

        if config.USE_VIRTUAL_BASES_OPT:
            for d_node in data.D:
                m.addConstr(x[self.inst.d_to_lv[d_node]] <= x[d_node], name=f"link_virtual_{d_node}")

        fixed_cost_expr = gp.quicksum(self.inst.f[n_node] * x[n_node] for n_node in data.N)
        min_wait_expr = gp.quicksum(
            data.service_wait[s_idx] * u[l_node, s_idx]
            for l_node in data.L
            for s_idx in data.service_wait
        )

        m.addConstr(
            fixed_cost_expr + config.COST_DRONE * min_wait_expr <= config.BUDGET,
            name="budget_lower_bound",
        )
        m.addConstr(
            min_wait_expr <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[l_node] for l_node in data.L_hat),
            name="fleet_capacity_lower_bound",
        )
        m.addConstr(
            theta <= gp.quicksum(data.H.get(i_node, 0.0) * x[i_node] for i_node in data.P),
            name="theta_pickup_open",
        )
        m.addConstr(
            theta <= gp.quicksum(data.delivery_demand.get(j_node, 0.0) * x[j_node] for j_node in data.D),
            name="theta_delivery_open",
        )
        m.addConstr(
            theta <= data.total_demand * gp.quicksum(x[l_node] for l_node in data.L_hat),
            name="theta_needs_base",
        )

        self._master = m
        self._master_x = x
        self._master_u = u
        self._master_theta = theta

    def _extract_master_solution(self):
        x_hat = {}
        for node in self.data.N + self.data.L_v:
            x_hat[node] = 1 if self._master_x[node].X >= 0.5 else 0

        u_hat = {}
        for l_node in self.data.L:
            for s_idx in self.data.S:
                u_hat[(l_node, s_idx)] = 1 if self._master_u[l_node, s_idx].X >= 0.5 else 0

        return x_hat, u_hat

    def _extract_callback_solution(self, model):
        x_hat = {}
        for node in self.data.N + self.data.L_v:
            x_hat[node] = 1 if model.cbGetSolution(self._master_x[node]) >= 0.5 else 0

        u_hat = {}
        for l_node in self.data.L:
            for s_idx in self.data.S:
                u_hat[(l_node, s_idx)] = 1 if model.cbGetSolution(self._master_u[l_node, s_idx]) >= 0.5 else 0

        theta_hat = model.cbGetSolution(self._master_theta)
        return x_hat, u_hat, theta_hat

    def _design_signature(self, x_hat, u_hat):
        return (
            tuple(x_hat[node] for node in self.data.N + self.data.L_v),
            tuple(u_hat[(l_node, s_idx)] for l_node in self.data.L for s_idx in self.data.S),
        )

    def _build_optimality_expr(self, x_hat, u_hat, subproblem):
        """Benders optimality cut: theta <= Z* + sum_i pi_i (b_i(x,u) - b_i(x_hat,u_hat))."""
        Z_star = subproblem["obj"]
        duals = subproblem["dual_terms"]
        data = self.data

        expr = gp.LinExpr(Z_star)

        # alpha1: sum_l y_li <= x_hat_i
        for i_node, dv in duals["alpha1"].items():
            expr += dv * (self._master_x[i_node] - x_hat[i_node])

        # alpha2: y_li <= x_hat_l
        for (l_node, i_node), dv in duals["alpha2"].items():
            expr += dv * (self._master_x[l_node] - x_hat[l_node])

        # alpha4: w_lij <= x_hat_j
        for (l_node, i_node, j_node), dv in duals["alpha4"].items():
            expr += dv * (self._master_x[j_node] - x_hat[j_node])

        # alpha5: y_li - w_lij <= 1 - x_hat_j  (RHS sensitivity flips sign)
        for (l_node, i_node, j_node), dv in duals["alpha5"].items():
            expr -= dv * (self._master_x[j_node] - x_hat[j_node])

        # epsilon3: z_lis <= H_i * eta_hat_ls
        for (l_node, i_node, s_idx), dv in duals["epsilon3"].items():
            H_i = data.H.get(i_node, 0.0)
            expr += dv * H_i * (self._master_u[l_node, s_idx] - u_hat[(l_node, s_idx)])

        # epsilon4: D_li - z_lis <= H_i*(1 - eta_hat_ls)  (RHS sensitivity flips sign)
        for (l_node, i_node, s_idx), dv in duals["epsilon4"].items():
            H_i = data.H.get(i_node, 0.0)
            expr -= dv * H_i * (self._master_u[l_node, s_idx] - u_hat[(l_node, s_idx)])

        # pi1: -MAX_WAIT_LV * sum psi <= -Q_hat  (RHS = -Q)
        for lv_node, dv in duals["pi1"].items():
            Q_master = gp.quicksum(
                data.service_wait.get(s_idx, 0.0) * self._master_u[lv_node, s_idx]
                for s_idx in data.S
            )
            Q_hat = sum(
                data.service_wait.get(s_idx, 0.0) * u_hat[(lv_node, s_idx)]
                for s_idx in data.S
            )
            expr -= dv * (Q_master - Q_hat)

        # pi2: sum psi <= limit * x_hat_u
        for nd, dv in duals["pi2"].items():
            qos_limit = config.MAX_WAITING_FACTOR_P if nd in data.P else config.MAX_WAITING_FACTOR_D
            expr += dv * qos_limit * (self._master_x[nd] - x_hat[nd])

        # pi3: sum phi^k <= U_k * x_hat_k
        for k_node, dv in duals["pi3"].items():
            expr += dv * config.MAX_FLOW_E * (self._master_x[k_node] - x_hat[k_node])

        # sigma1: transit - cap*replen <= -Q_hat  (RHS = -Q)
        for lv_node, dv in duals["sigma1"].items():
            Q_master = gp.quicksum(
                data.service_wait.get(s_idx, 0.0) * self._master_u[lv_node, s_idx]
                for s_idx in data.S
            )
            Q_hat = sum(
                data.service_wait.get(s_idx, 0.0) * u_hat[(lv_node, s_idx)]
                for s_idx in data.S
            )
            expr -= dv * (Q_master - Q_hat)

        # kappa1: fleet - gamma <= -sum Q_hat  (RHS = -sum Q)
        kappa1_val = duals["kappa1"]
        total_Q_master = gp.quicksum(
            data.service_wait.get(s_idx, 0.0) * self._master_u[l_node, s_idx]
            for l_node in data.L
            for s_idx in data.S
        )
        total_Q_hat = sum(
            data.service_wait.get(s_idx, 0.0) * u_hat[(l_node, s_idx)]
            for l_node in data.L
            for s_idx in data.S
        )
        expr -= kappa1_val * (total_Q_master - total_Q_hat)

        # kappa2: c*gamma <= C - sum f*x_hat  (RHS = C - sum f*x)
        kappa2_val = duals["kappa2"]
        for n_node in data.N:
            expr -= kappa2_val * self.inst.f[n_node] * (self._master_x[n_node] - x_hat[n_node])

        # kappa3: gamma <= V * sum x_hat_l  (RHS = V * sum x_l)
        kappa3_val = duals["kappa3"]
        for l_node in data.L_hat:
            expr += kappa3_val * config.MAX_FLEET_PER_BASE * (self._master_x[l_node] - x_hat[l_node])

        return expr

    def _compute_queue_loads(self, u_hat):
        return {
            l_node: sum(
                self.data.service_wait.get(s_idx, 0.0) * u_hat[(l_node, s_idx)]
                for s_idx in self.data.S
            )
            for l_node in self.data.L
        }

    def _compute_base_pickup_coeffs(self, x_hat, u_hat):
        rho_hat = {
            l_node: sum(self.data.rho[s_idx] * u_hat[(l_node, s_idx)] for s_idx in self.data.S)
            for l_node in self.data.L
        }
        pickup_open_demand = {
            i_node: sum(
                self.inst.h.get((i_node, j_node), 0.0) * x_hat[j_node]
                for j_node in self.data.D
                if (i_node, j_node) in self.data.A_PD_set
            )
            for i_node in self.data.P
        }
        return {
            (l_node, i_node): rho_hat[l_node] * pickup_open_demand[i_node]
            for l_node, i_node in self.data.A_LP
        }

    def _build_incumbent_expr(self, x_hat, u_hat, sub_obj):
        mismatch_terms = []

        for node in self.data.N + self.data.L_v:
            mismatch_terms.append((1 - self._master_x[node]) if x_hat[node] == 1 else self._master_x[node])
        for l_node in self.data.L:
            for s_idx in self.data.S:
                mismatch_terms.append(
                    (1 - self._master_u[l_node, s_idx])
                    if u_hat[(l_node, s_idx)] == 1
                    else self._master_u[l_node, s_idx]
                )
        return sub_obj + self.data.total_demand * gp.quicksum(mismatch_terms)

    def _build_nogood_expr(self, x_hat, u_hat):
        terms = []

        for node in self.data.N + self.data.L_v:
            terms.append((1 - self._master_x[node]) if x_hat[node] == 1 else self._master_x[node])
        for l_node in self.data.L:
            for s_idx in self.data.S:
                terms.append(
                    (1 - self._master_u[l_node, s_idx])
                    if u_hat[(l_node, s_idx)] == 1
                    else self._master_u[l_node, s_idx]
                )
        return gp.quicksum(terms) >= 1

    def _build_subproblem(self, x_hat, u_hat):
        data = self.data
        inst = self.inst
        q_hat = self._compute_queue_loads(u_hat)

        sp = gp.Model("MLND_Benders_Subproblem")
        sp.setParam("OutputFlag", 0)
        sp.setParam("QCPDual", 1)
        sp.setParam("DualReductions", 0)


        # --- Variables ---
        y = sp.addVars(data.A_LP, lb=0.0, ub=1.0, name="y")
        w_keys = [
            (l_node, i_node, j_node)
            for l_node, i_node in data.A_LP
            for j_node in data.D
            if (i_node, j_node) in data.A_PD_set
        ]
        w = sp.addVars(w_keys, lb=0.0, name="w")
        D_var = sp.addVars(data.A_LP, lb=0.0, name="D")
        z_keys = [
            (l_node, i_node, s_idx)
            for l_node, i_node in data.A_LP
            for s_idx in data.S
        ]
        z = sp.addVars(z_keys, lb=0.0, name="z")
        psi = sp.addVars(data.A, lb=0.0, name="psi")
        phi = sp.addVars(data.phi_keys, lb=0.0, name="phi")
        xi = sp.addVars(data.P + data.D, lb=0.0, ub=10000.0, name="xi")
        lam = sp.addVars(
            data.P + data.D,
            lb=0.0,
            ub={nd: inst.mu[nd] - 1e-3 for nd in data.P + data.D},
            name="lam",
        )
        gamma = sp.addVar(lb=0.0, ub=100000.0, name="gamma")

        # Relaxation slacks to guarantee feasibility
        slack_pi1 = sp.addVars(data.L_v, lb=0.0, name="slack_pi1")
        slack_sigma1 = sp.addVars(data.L_v, lb=0.0, name="slack_sigma1")
        slack_kappa1 = sp.addVar(lb=0.0, name="slack_kappa1")
        slack_kappa2 = sp.addVar(lb=0.0, name="slack_kappa2")
        slack_kappa3 = sp.addVar(lb=0.0, name="slack_kappa3")

        PENALTY = 1e5
        sp.setObjective(
            gp.quicksum(psi[i_node, j_node] for i_node, j_node in data.A_PD)
            - PENALTY * (
                gp.quicksum(slack_pi1[lv] for lv in data.L_v)
                + gp.quicksum(slack_sigma1[lv] for lv in data.L_v)
                + slack_kappa1 + slack_kappa2 + slack_kappa3
            ),
            GRB.MAXIMIZE,
        )

        # --- Constraint alpha1: sum_l y_li <= x_hat_i ---
        for i_node in data.P:
            ls = [l_node for l_node in data.L if (l_node, i_node) in data.A_LP_set]
            if ls:
                sp.addConstr(
                    gp.quicksum(y[l_node, i_node] for l_node in ls) <= x_hat[i_node],
                    name=f"alpha1_{i_node}",
                )

        # --- Constraint alpha2: y_li <= x_hat_l ---
        for l_node, i_node in data.A_LP:
            sp.addConstr(y[l_node, i_node] <= x_hat[l_node], name=f"alpha2_{l_node}_{i_node}")

        # --- Constraint alpha3: w_lij <= y_li ---
        for l_node, i_node, j_node in w_keys:
            sp.addConstr(
                w[l_node, i_node, j_node] <= y[l_node, i_node],
                name=f"alpha3_{l_node}_{i_node}_{j_node}",
            )

        # --- Constraint alpha4: w_lij <= x_hat_j ---
        for l_node, i_node, j_node in w_keys:
            sp.addConstr(
                w[l_node, i_node, j_node] <= x_hat[j_node],
                name=f"alpha4_{l_node}_{i_node}_{j_node}",
            )

        # --- Constraint alpha5: y_li - w_lij <= 1 - x_hat_j ---
        for l_node, i_node, j_node in w_keys:
            sp.addConstr(
                y[l_node, i_node] - w[l_node, i_node, j_node] <= 1 - x_hat[j_node],
                name=f"alpha5_{l_node}_{i_node}_{j_node}",
            )

        # --- Constraint alpha6: D_li = sum_j h_ij * w_lij ---
        for l_node, i_node in data.A_LP:
            sp.addConstr(
                D_var[l_node, i_node]
                == gp.quicksum(
                    inst.h.get((i_node, j_node), 0.0) * w[l_node, i_node, j_node]
                    for j_node in data.D
                    if (i_node, j_node) in data.A_PD_set
                ),
                name=f"alpha6_{l_node}_{i_node}",
            )

        # --- Constraint epsilon2: z_lis <= D_li ---
        for l_node, i_node, s_idx in z_keys:
            sp.addConstr(
                z[l_node, i_node, s_idx] <= D_var[l_node, i_node],
                name=f"epsilon2_{l_node}_{i_node}_{s_idx}",
            )

        # --- Constraint epsilon3: z_lis <= H_i * eta_hat_ls ---
        for l_node, i_node, s_idx in z_keys:
            H_i = data.H.get(i_node, 0.0)
            sp.addConstr(
                z[l_node, i_node, s_idx] <= H_i * u_hat[(l_node, s_idx)],
                name=f"epsilon3_{l_node}_{i_node}_{s_idx}",
            )

        # --- Constraint epsilon4: D_li - z_lis <= H_i*(1 - eta_hat_ls) ---
        for l_node, i_node, s_idx in z_keys:
            H_i = data.H.get(i_node, 0.0)
            sp.addConstr(
                D_var[l_node, i_node] - z[l_node, i_node, s_idx]
                <= H_i * (1 - u_hat[(l_node, s_idx)]),
                name=f"epsilon4_{l_node}_{i_node}_{s_idx}",
            )

        # --- Constraint epsilon5: psi_li = sum_s rho_s * z_lis ---
        for l_node, i_node in data.A_LP:
            sp.addConstr(
                psi[l_node, i_node]
                == gp.quicksum(
                    data.rho[s_idx] * z[l_node, i_node, s_idx] for s_idx in data.S
                ),
                name=f"epsilon5_{l_node}_{i_node}",
            )

        # --- Constraint beta1: flow conservation ---
        for v_node in data.N + data.L_v:
            sp.addConstr(
                gp.quicksum(psi[arc] for arc in data.in_arcs.get(v_node, []))
                == gp.quicksum(psi[arc] for arc in data.out_arcs.get(v_node, [])),
                name=f"beta1_{v_node}",
            )

        # --- Constraint beta2: demand bound ---
        for i_node, j_node in data.A_PD:
            sp.addConstr(psi[i_node, j_node] <= inst.h[i_node, j_node], name=f"beta2_{i_node}_{j_node}")

        # --- Constraint beta3: flow decomposition ---
        for u_node, v_node in data.A:
            sp.addConstr(
                psi[u_node, v_node]
                == gp.quicksum(phi[u_node, v_node, k_node] for k_node in data.phi_for_arc.get((u_node, v_node), [])),
                name=f"beta3_{u_node}_{v_node}",
            )

        # --- Constraint pi2: QoS ---
        for nd in data.P + data.D:
            qos_limit = config.MAX_WAITING_FACTOR_P if nd in data.P else config.MAX_WAITING_FACTOR_D
            sp.addConstr(
                gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, [])) <= qos_limit * x_hat[nd],
                name=f"pi2_{nd}",
            )

        # --- Constraint pi3: swap station capacity ---
        for k_node in data.E:
            sp.addConstr(
                gp.quicksum(phi[key] for key in data.phi_through_swap.get(k_node, [])) <= config.MAX_FLOW_E * x_hat[k_node],
                name=f"pi3_{k_node}",
            )

        # --- Constraint pi1: virtual base wait ---
        if config.USE_VIRTUAL_BASES_OPT:
            for lv_node in data.L_v:
                sp.addConstr(
                    -config.MAX_WAITING_FACTOR_LV * gp.quicksum(
                        psi[u_node, i_node] for u_node, i_node in data.A_LP if u_node == lv_node
                    ) - slack_pi1[lv_node] <= -q_hat[lv_node],
                    name=f"pi1_{lv_node}",
                )

        # --- Constraint delta: SOC queueing (Linear Outer Approximation) ---
        for nd in data.P + data.D:
            sp.addConstr(lam[nd] == gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, [])), name=f"lambda_def_{nd}")
            
            for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                lam0 = inst.mu[nd] * frac
                den = inst.mu[nd] - lam0
                f0 = lam0 / den
                fp = inst.mu[nd] / (den * den)
                sp.addConstr(xi[nd] >= (f0 - fp * lam0) + fp * lam[nd], name=f"queue_oa_{nd}_{frac}")

        # --- Constraint sigma1: energy virtual base ---
        if config.USE_VIRTUAL_BASES_OPT:
            for lv_node in data.L_v:
                transit = gp.LinExpr()
                replen = gp.LinExpr()
                for u_node, i_node in data.A_LP:
                    if u_node != lv_node:
                        continue
                    if (u_node, i_node, "0") in data.phi_keys_set:
                        transit += phi[u_node, i_node, "0"] * inst.get_time(u_node, i_node)
                    for k_node in data.E_uv.get((u_node, i_node), []):
                        if (u_node, i_node, k_node) in data.phi_keys_set:
                            travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, i_node)
                            transit += phi[u_node, i_node, k_node] * travel_time
                            replen += phi[u_node, i_node, k_node]

                sp.addConstr(
                    transit - (data.cap_eff / config.ENERGY_RATE_OMEGA) * replen - slack_sigma1[lv_node] <= -q_hat[lv_node],
                    name=f"sigma1_{lv_node}",
                )

        # --- Constraint sigma2: energy pickup ---
        for i_node in data.P:
            transit = gp.LinExpr()
            replen = gp.LinExpr()
            for u_node, v_node in data.A_PD:
                if u_node != i_node:
                    continue
                if (u_node, v_node, "0") in data.phi_keys_set:
                    transit += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
                for k_node in data.E_uv.get((u_node, v_node), []):
                    if (u_node, v_node, k_node) in data.phi_keys_set:
                        travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                        transit += phi[u_node, v_node, k_node] * travel_time
                        replen += phi[u_node, v_node, k_node]
            sp.addConstr(
                xi[i_node] + transit - (data.cap_eff / config.ENERGY_RATE_OMEGA) * replen <= 0,
                name=f"sigma2_{i_node}",
            )

        # --- Constraint kappa1: fleet sizing ---
        fleet_expr = gp.quicksum(xi[nd] for nd in data.P + data.D)
        for u_node, v_node in data.A:
            if (u_node, v_node, "0") in data.phi_keys_set:
                fleet_expr += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
            for k_node in data.E_uv.get((u_node, v_node), []):
                if (u_node, v_node, k_node) in data.phi_keys_set:
                    travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                    fleet_expr += phi[u_node, v_node, k_node] * travel_time

        total_q_hat = sum(q_hat.values())
        sp.addConstr(fleet_expr - gamma - slack_kappa1 <= -total_q_hat, name="kappa1")

        # --- Constraint kappa2: budget ---
        budget_rhs = config.BUDGET - sum(inst.f[n_node] * x_hat[n_node] for n_node in data.N)
        sp.addConstr(config.COST_DRONE * gamma - slack_kappa2 <= budget_rhs, name="kappa2")

        # --- Constraint kappa3: fleet limit ---
        sp.addConstr(
            gamma - slack_kappa3 <= config.MAX_FLEET_PER_BASE * sum(x_hat[l_node] for l_node in data.L_hat),
            name="kappa3",
        )

        sp.update()
        return sp, {
            "psi": psi,
            "gamma": gamma,
            "w_keys": w_keys,
            "z_keys": z_keys,
        }

    def _solve_subproblem(self, x_hat, u_hat):
        sp, vars_map = self._build_subproblem(x_hat, u_hat)
        sp.optimize()

        result = {
            "status": sp.Status,
            "model": sp,
            "psi": vars_map["psi"],
            "gamma": vars_map["gamma"],
        }

        if sp.Status == GRB.INFEASIBLE and len(self.history) == 2:
            sp.computeIIS()
            sp.write("infeasible_2.ilp")

        if sp.Status == GRB.OPTIMAL:
            result["obj"] = sp.ObjVal
            if len(self.history) == 1:
                sp.write("subproblem_fail.mps")

            open("debug_missing_duals.txt", "a").write(f"\nIsMIP: {sp.IsMIP}, Status: {sp.Status}\n")
            duals_available = True
            dual_terms = {
                "alpha1": {},
                "alpha2": {},
                "alpha4": {},
                "alpha5": {},
                "epsilon3": {},
                "epsilon4": {},
                "pi1": {},
                "pi2": {},
                "pi3": {},
                "sigma1": {},
                "kappa1": 0.0,
                "kappa2": 0.0,
                "kappa3": 0.0,
            }

            for i_node in self.data.P:
                constr = sp.getConstrByName(f"alpha1_{i_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["alpha1"][i_node] = dual_val

            for l_node, i_node in self.data.A_LP:
                constr = sp.getConstrByName(f"alpha2_{l_node}_{i_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["alpha2"][(l_node, i_node)] = dual_val

            for l_node, i_node, j_node in vars_map["w_keys"]:
                constr = sp.getConstrByName(f"alpha4_{l_node}_{i_node}_{j_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["alpha4"][(l_node, i_node, j_node)] = dual_val

            for l_node, i_node, j_node in vars_map["w_keys"]:
                constr = sp.getConstrByName(f"alpha5_{l_node}_{i_node}_{j_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["alpha5"][(l_node, i_node, j_node)] = dual_val

            for l_node, i_node, s_idx in vars_map["z_keys"]:
                constr = sp.getConstrByName(f"epsilon3_{l_node}_{i_node}_{s_idx}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["epsilon3"][(l_node, i_node, s_idx)] = dual_val

            for l_node, i_node, s_idx in vars_map["z_keys"]:
                constr = sp.getConstrByName(f"epsilon4_{l_node}_{i_node}_{s_idx}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["epsilon4"][(l_node, i_node, s_idx)] = dual_val

            for lv_node in self.data.L_v:
                constr = sp.getConstrByName(f"pi1_{lv_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        print(f"Missing dual for pi1_{lv_node}")
                        duals_available = False
                    else:
                        dual_terms["pi1"][lv_node] = dual_val

            for nd in self.data.P + self.data.D:
                constr = sp.getConstrByName(f"pi2_{nd}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["pi2"][nd] = dual_val

            for k_node in self.data.E:
                constr = sp.getConstrByName(f"pi3_{k_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["pi3"][k_node] = dual_val

            for lv_node in self.data.L_v:
                constr = sp.getConstrByName(f"sigma1_{lv_node}")
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms["sigma1"][lv_node] = dual_val

            for key in ("kappa1", "kappa2", "kappa3"):
                constr = sp.getConstrByName(key)
                if constr is not None:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                    else:
                        dual_terms[key] = dual_val

            all_duals_zero = True
            for k, val in dual_terms.items():
                if isinstance(val, dict):
                    if any(abs(v) > 1e-6 for v in val.values()):
                        all_duals_zero = False
                elif abs(val) > 1e-6:
                    all_duals_zero = False

            print(f"DEBUG: obj={result["obj"]}, all_duals_zero={all_duals_zero}, dict_len={len(dual_terms)}")
            if abs(result["obj"]) <= 1e-6:
                print("DEBUG: DUALS:")
                for k,v in dual_terms.items():
                    if isinstance(v, dict) and any(abs(val) > 1e-6 for val in v.values()):
                        print(f" {k} -> max {max(abs(val) for val in v.values())}")
                    elif not isinstance(v, dict) and abs(v) > 1e-6:
                        print(f" {k} -> {v}")

            if abs(result["obj"]) <= 1e-6 and all_duals_zero:
                # Degenerate flat cut that would bound theta <= 0 incorrectly.
                # Treat as infeasible to emit a nogood/incumbent cut instead.
                return {"status": GRB.INFEASIBLE, "cut": "nogood"}

            result["dual_terms"] = dual_terms
            result["duals_available"] = duals_available
            result["served_arcs"] = {
                arc: vars_map["psi"][arc].X
                for arc in self.data.A_PD
                if vars_map["psi"][arc].X > 1e-6
            }
            result["gamma_val"] = vars_map["gamma"].X

        return result

    @staticmethod
    @staticmethod
    def _get_linear_dual(constr):
        if constr is None:
            return 0.0
        for attr_name in ("Pi", "QCPi", "BarPi"):
            try:
                val = constr.getAttr(attr_name)
                return val
            except Exception as e:
                open('debug_missing_duals.txt', 'a').write(f"Exception for {constr.ConstrName}: {e}\n")
                pass
        return 0.0

    @staticmethod
    def _get_quadratic_dual(qconstr):
        try:
            return qconstr.getAttr("QCPi")
        except Exception:
            return None

    def _add_optimality_cut(self, x_hat, u_hat, subproblem):
        expr = self._build_optimality_expr(x_hat, u_hat, subproblem)
        self._master.addConstr(self._master_theta <= expr, name=f"benders_cut_{len(self.history)}")

    def _add_incumbent_cut(self, x_hat, u_hat, sub_obj):
        expr = self._build_incumbent_expr(x_hat, u_hat, sub_obj)
        self._master.addConstr(self._master_theta <= expr, name=f"incumbent_cut_{len(self.history)}")

    def _add_no_good_cut(self, x_hat, u_hat):
        self._master.addConstr(self._build_nogood_expr(x_hat, u_hat), name=f"nogood_{len(self.history)}")

    def _callback(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        elapsed = time.time() - self._solve_start
        if elapsed >= self.time_limit:
            model.terminate()
            return

        x_hat, u_hat, theta_hat = self._extract_callback_solution(model)
        signature = self._design_signature(x_hat, u_hat)
        if signature in self._evaluated_designs:
            return
        self._evaluated_designs.add(signature)

        self.callback_stats["incumbents"] += 1
        if self.max_iter and self.callback_stats["incumbents"] > self.max_iter:
            model.terminate()
            return

        subproblem = self._solve_subproblem(x_hat, u_hat)
        status = subproblem["status"]
        record = {
            "iter": self.callback_stats["incumbents"],
            "theta": theta_hat,
            "sub_status": status,
            "time": elapsed,
        }

        if status == GRB.OPTIMAL:
            sub_obj = subproblem["obj"]
            record["sub_obj"] = sub_obj
            record["gap"] = theta_hat - sub_obj
            record["duals_available"] = subproblem.get("duals_available", False)

            if self.best_obj is None or sub_obj > self.best_obj + 1e-9:
                self.best_obj = sub_obj
                self.best_solution = {
                    "objective": sub_obj,
                    "x": dict(x_hat),
                    "u": dict(u_hat),
                    "gamma": subproblem["gamma_val"],
                    "served_arcs": subproblem["served_arcs"],
                    "history": self.history,
                }

            if theta_hat > sub_obj + self.gap_tol:
                if subproblem.get("duals_available", False):
                    model.cbLazy(self._master_theta <= self._build_optimality_expr(x_hat, u_hat, subproblem))
                    self.callback_stats["optimality"] += 1
                    record["cut"] = "optimality"
                else:
                    model.cbLazy(self._master_theta <= self._build_incumbent_expr(x_hat, u_hat, sub_obj))
                    self.callback_stats["incumbent"] += 1
                    record["cut"] = "incumbent"
            else:
                record["cut"] = "none"

        else:
            model.cbLazy(self._build_nogood_expr(x_hat, u_hat))
            self.callback_stats["nogood"] += 1
            record["cut"] = "nogood"

        self.history.append(record)

        if self.verbose:
            if status == GRB.OPTIMAL:
                print(
                    f"Inc {record['iter']:3d} | Theta={theta_hat:10.4f} | Sub={record['sub_obj']:10.4f} | "
                    f"Gap={record['gap']:10.6f} | Cut={record['cut']}"
                )
            else:
                print(
                    f"Inc {record['iter']:3d} | Theta={theta_hat:10.4f} | SubStatus={status} | Cut=nogood"
                )

    def solve(self):
        start = time.time()
        self._solve_start = start
        self.history = []
        self.best_obj = None
        self.best_master_obj = None
        self.best_solution = None
        self.callback_stats = {"incumbents": 0, "optimality": 0, "incumbent": 0, "nogood": 0}
        self._evaluated_designs = set()
        self._build_master()
        self._master.setParam("LazyConstraints", 1)
        self._master.setParam("TimeLimit", self.time_limit)
        self._master.setParam("OutputFlag", 1 if self.verbose else 0)

        if self.verbose:
            print("=" * 72)
            print("CALLBACK BENDERS DECOMPOSITION FOR MLND MODEL")
            print("=" * 72)
            print("Master: binary design variables only; lazy cuts at incumbent solutions")

        self._master.optimize(self._callback)

        if self._master.SolCount > 0:
            self.best_master_obj = self._master.ObjVal

        if self.best_solution is not None:
            self.best_solution["history"] = self.history

        if self.verbose:
            print("=" * 72)
            if self.best_obj is None:
                print("No feasible Benders incumbent found.")
            else:
                print(f"Done  best={self.best_obj:.4f}  time={time.time() - start:.1f}s")
            print("=" * 72)

        return self.best_obj

    def get_solution(self):
        return self.best_solution


# Backward-compatible alias for the previous implementation, now named correctly.
OABendersDecomposition = QueueOuterApproximation