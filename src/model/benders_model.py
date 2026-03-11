"""Decomposition algorithms for the MLND model.

This module now distinguishes between:

* ``QueueOuterApproximation``: the previous OA algorithm that iteratively
  tightens tangent cuts for the convex queueing term.
* ``BendersDecomposition``: a binary-master / continuous-subproblem Benders
  decomposition. The subproblem is solved in primal form as a convex QCP,
  and master cuts are generated from subproblem duals.
"""

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
        obj_hat = subproblem["obj"]
        grad_x = subproblem["grad_x"]
        grad_u = subproblem["grad_u"]

        constant = obj_hat
        constant -= sum(grad_x[node] * x_hat[node] for node in grad_x)
        constant -= sum(grad_u[key] * u_hat[key] for key in grad_u)

        expr = gp.LinExpr(constant)
        for node, coeff in grad_x.items():
            if abs(coeff) > 1e-9:
                expr += coeff * self._master_x[node]
        for (l_node, s_idx), coeff in grad_u.items():
            if abs(coeff) > 1e-9:
                expr += coeff * self._master_u[l_node, s_idx]
        return expr

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

        sp = gp.Model("MLND_Benders_Subproblem")
        sp.setParam("OutputFlag", 0)
        sp.setParam("QCPDual", 1)
        sp.setParam("Method", 2)
        sp.setParam("Crossover", 0)
        sp.setParam("Presolve", 0)

        y = sp.addVars(data.A_LP, lb=0.0, ub=1.0, name="y")
        w_keys = [(l_node, i_node, j_node) for l_node, i_node in data.A_LP for j_node in data.D if (i_node, j_node) in data.A_PD_set]
        w = sp.addVars(w_keys, lb=0.0, name="w")
        D_var = sp.addVars(data.A_LP, lb=0.0, name="D")
        z_keys = [(l_node, i_node, s_idx) for l_node, i_node in data.A_LP for s_idx in data.S]
        z = sp.addVars(z_keys, lb=0.0, name="z")
        psi = sp.addVars(data.A, lb=0.0, name="psi")
        phi = sp.addVars(data.phi_keys, lb=0.0, name="phi")
        xi = sp.addVars(data.P + data.D, lb=0.0, name="xi")
        lam = sp.addVars(
            data.P + data.D,
            lb=0.0,
            ub={nd: inst.mu[nd] - 1e-3 for nd in data.P + data.D},
            name="lam",
        )
        queue_soc_rhs = sp.addVars(data.P + data.D, lb=0.0, name="queue_soc_rhs")
        queue_soc_lhs = sp.addVars(data.P + data.D, lb=-GRB.INFINITY, name="queue_soc_lhs")
        gamma = sp.addVar(lb=0.0, name="gamma")

        sp.setObjective(gp.quicksum(psi[i_node, j_node] for i_node, j_node in data.A_PD), GRB.MAXIMIZE)

        cut_data = {"x": {}, "u": {}}

        for i_node in data.P:
            ls = [l_node for l_node in data.L if (l_node, i_node) in data.A_LP_set]
            if ls:
                constr = sp.addConstr(
                    gp.quicksum(y[l_node, i_node] for l_node in ls) <= x_hat[i_node],
                    name=f"assign_pickup_{i_node}",
                )
                cut_data["x"].setdefault(i_node, []).append((constr, 1.0))

        for l_node, i_node in data.A_LP:
            constr = sp.addConstr(y[l_node, i_node] <= x_hat[l_node], name=f"assign_base_{l_node}_{i_node}")
            cut_data["x"].setdefault(l_node, []).append((constr, 1.0))

        for l_node, i_node in data.A_LP:
            H_i = data.H.get(i_node, 0.0)
            for j_node in data.D:
                if (i_node, j_node) not in data.A_PD_set:
                    continue
                constr = sp.addConstr(w[l_node, i_node, j_node] <= x_hat[j_node], name=f"w_cap_open_{l_node}_{i_node}_{j_node}")
                cut_data["x"].setdefault(j_node, []).append((constr, 1.0))
                sp.addConstr(w[l_node, i_node, j_node] <= y[l_node, i_node], name=f"w_cap_y_{l_node}_{i_node}_{j_node}")
                constr = sp.addConstr(
                    -w[l_node, i_node, j_node] + y[l_node, i_node] <= 1 - x_hat[j_node],
                    name=f"w_link_{l_node}_{i_node}_{j_node}",
                )
                cut_data["x"].setdefault(j_node, []).append((constr, -1.0))

            sp.addConstr(
                D_var[l_node, i_node]
                == gp.quicksum(
                    inst.h.get((i_node, j_node), 0.0) * w[l_node, i_node, j_node]
                    for j_node in data.D
                    if (i_node, j_node) in data.A_PD_set
                ),
                name=f"demand_agg_{l_node}_{i_node}",
            )

            for s_idx in data.S:
                sp.addConstr(z[l_node, i_node, s_idx] <= D_var[l_node, i_node], name=f"z_cap_D_{l_node}_{i_node}_{s_idx}")
                constr = sp.addConstr(
                    z[l_node, i_node, s_idx] <= H_i * u_hat[(l_node, s_idx)],
                    name=f"z_cap_u_{l_node}_{i_node}_{s_idx}",
                )
                cut_data["u"].setdefault((l_node, s_idx), []).append((constr, H_i))
                constr = sp.addConstr(
                    -z[l_node, i_node, s_idx] + D_var[l_node, i_node] <= H_i * (1 - u_hat[(l_node, s_idx)]),
                    name=f"z_link_{l_node}_{i_node}_{s_idx}",
                )
                cut_data["u"].setdefault((l_node, s_idx), []).append((constr, H_i))

            sp.addConstr(
                psi[l_node, i_node] == gp.quicksum(data.rho[s_idx] * z[l_node, i_node, s_idx] for s_idx in data.S),
                name=f"base_pickup_flow_{l_node}_{i_node}",
            )

        for v_node in data.N + data.L_v:
            sp.addConstr(
                gp.quicksum(psi[arc] for arc in data.in_arcs.get(v_node, []))
                == gp.quicksum(psi[arc] for arc in data.out_arcs.get(v_node, [])),
                name=f"flow_cons_{v_node}",
            )

        for i_node, j_node in data.A_PD:
            sp.addConstr(psi[i_node, j_node] <= inst.h[i_node, j_node], name=f"demand_bound_{i_node}_{j_node}")

        for u_node, v_node in data.A:
            sp.addConstr(
                psi[u_node, v_node]
                == gp.quicksum(phi[u_node, v_node, k_node] for k_node in data.phi_for_arc.get((u_node, v_node), [])),
                name=f"flow_decomp_{u_node}_{v_node}",
            )

        for nd in data.P + data.D:
            qos_limit = config.MAX_WAITING_FACTOR_P if nd in data.P else config.MAX_WAITING_FACTOR_D
            constr = sp.addConstr(
                gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, [])) <= qos_limit * x_hat[nd],
                name=f"qos_{nd}",
            )
            cut_data["x"].setdefault(nd, []).append((constr, qos_limit))

        for k_node in data.E:
            constr = sp.addConstr(
                gp.quicksum(phi[key] for key in data.phi_through_swap.get(k_node, [])) <= config.MAX_FLOW_E * x_hat[k_node],
                name=f"swap_open_{k_node}",
            )
            cut_data["x"].setdefault(k_node, []).append((constr, config.MAX_FLOW_E))

        if config.USE_VIRTUAL_BASES_OPT:
            for lv_node in data.L_v:
                rhs = -sum(data.service_wait.get(s_idx, 0.0) * u_hat[(lv_node, s_idx)] for s_idx in data.S)
                constr = sp.addConstr(
                    -config.MAX_WAITING_FACTOR_LV * gp.quicksum(
                        psi[u_node, i_node] for u_node, i_node in data.A_LP if u_node == lv_node
                    ) <= rhs,
                    name=f"virtual_wait_{lv_node}",
                )
                for s_idx in data.service_wait:
                    cut_data["u"].setdefault((lv_node, s_idx), []).append((constr, -data.service_wait[s_idx]))

        for nd in data.P + data.D:
            sp.addConstr(lam[nd] == gp.quicksum(psi[arc] for arc in data.out_arcs.get(nd, [])), name=f"lambda_def_{nd}")
            sp.addConstr(
                queue_soc_rhs[nd] == xi[nd] - lam[nd] + inst.mu[nd] + 1,
                name=f"queue_soc_rhs_def_{nd}",
            )
            sp.addConstr(
                queue_soc_lhs[nd] == xi[nd] + lam[nd] - inst.mu[nd] + 1,
                name=f"queue_soc_lhs_def_{nd}",
            )
            sp.addQConstr(
                4 * inst.mu[nd] + queue_soc_lhs[nd] * queue_soc_lhs[nd]
                <= queue_soc_rhs[nd] * queue_soc_rhs[nd],
                name=f"queue_soc_{nd}",
            )
            sp.addConstr(queue_soc_rhs[nd] >= 0, name=f"queue_soc_pos_{nd}")

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

                rhs = -config.ENERGY_RATE_OMEGA * sum(
                    data.service_wait.get(s_idx, 0.0) * u_hat[(lv_node, s_idx)]
                    for s_idx in data.S
                )
                constr = sp.addConstr(
                    config.ENERGY_RATE_OMEGA * transit - data.cap_eff * replen <= rhs,
                    name=f"energy_lv_{lv_node}",
                )
                for s_idx in data.service_wait:
                    cut_data["u"].setdefault((lv_node, s_idx), []).append(
                        (constr, -config.ENERGY_RATE_OMEGA * data.service_wait[s_idx])
                    )

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
                config.ENERGY_RATE_OMEGA * (xi[i_node] + transit) <= data.cap_eff * replen,
                name=f"energy_p_{i_node}",
            )

        # for j_node in data.D:
        #     transit = gp.LinExpr()
        #     replen = gp.LinExpr()
        #     for u_node, v_node in data.out_arcs.get(j_node, []):
        #         if (u_node, v_node, "0") in data.phi_keys_set:
        #             transit += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
        #             if v_node in data.L_hat_set:
        #                 replen += phi[u_node, v_node, "0"]
        #         for k_node in data.E_uv.get((u_node, v_node), []):
        #             if (u_node, v_node, k_node) in data.phi_keys_set:
        #                 travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
        #                 transit += phi[u_node, v_node, k_node] * travel_time
        #                 replen += phi[u_node, v_node, k_node]
        #     sp.addConstr(
        #         config.ENERGY_RATE_OMEGA * (xi[j_node] + transit) <= data.cap_eff * replen,
        #         name=f"energy_d_{j_node}",
        #     )

        fleet_expr = gp.quicksum(xi[nd] for nd in data.P + data.D)
        for u_node, v_node in data.A:
            if (u_node, v_node, "0") in data.phi_keys_set:
                fleet_expr += phi[u_node, v_node, "0"] * inst.get_time(u_node, v_node)
            for k_node in data.E_uv.get((u_node, v_node), []):
                if (u_node, v_node, k_node) in data.phi_keys_set:
                    travel_time = inst.get_time(u_node, k_node) + inst.get_time(k_node, v_node)
                    fleet_expr += phi[u_node, v_node, k_node] * travel_time

        fleet_rhs = -sum(
            data.service_wait.get(s_idx, 0.0) * u_hat[(l_node, s_idx)]
            for l_node in data.L
            for s_idx in data.S
        )
        constr = sp.addConstr(fleet_expr - gamma <= fleet_rhs, name="fleet_calc")
        for l_node in data.L:
            for s_idx in data.service_wait:
                cut_data["u"].setdefault((l_node, s_idx), []).append((constr, -data.service_wait[s_idx]))

        budget_rhs = config.BUDGET - sum(inst.f[n_node] * x_hat[n_node] for n_node in data.N)
        constr = sp.addConstr(config.COST_DRONE * gamma <= budget_rhs, name="budget")
        for n_node in data.N:
            cut_data["x"].setdefault(n_node, []).append((constr, -inst.f[n_node]))

        constr = sp.addConstr(
            gamma <= config.MAX_FLEET_PER_BASE * sum(x_hat[l_node] for l_node in data.L_hat),
            name="gamma_limit",
        )
        for l_node in data.L_hat:
            cut_data["x"].setdefault(l_node, []).append((constr, config.MAX_FLEET_PER_BASE))

        sp.update()
        return sp, {"psi": psi, "gamma": gamma}, cut_data

    def _solve_subproblem(self, x_hat, u_hat):
        sp, vars_map, cut_data = self._build_subproblem(x_hat, u_hat)
        sp.optimize()

        result = {
            "status": sp.Status,
            "model": sp,
            "psi": vars_map["psi"],
            "gamma": vars_map["gamma"],
            "cut_data": cut_data,
        }

        if sp.Status == GRB.OPTIMAL:
            result["obj"] = sp.ObjVal
            gradients_x = {node: 0.0 for node in self.data.N + self.data.L_v}
            gradients_u = {(l_node, s_idx): 0.0 for l_node in self.data.L for s_idx in self.data.S}
            duals_available = True

            for node, entries in cut_data["x"].items():
                node_grad = 0.0
                for constr, coeff in entries:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                        break
                    node_grad += dual_val * coeff
                gradients_x[node] = node_grad
            for key, entries in cut_data["u"].items():
                key_grad = 0.0
                for constr, coeff in entries:
                    dual_val = self._get_linear_dual(constr)
                    if dual_val is None:
                        duals_available = False
                        break
                    key_grad += dual_val * coeff
                gradients_u[key] = key_grad

            result["grad_x"] = gradients_x
            result["grad_u"] = gradients_u
            result["duals_available"] = duals_available
            result["served_arcs"] = {
                arc: vars_map["psi"][arc].X
                for arc in self.data.A_PD
                if vars_map["psi"][arc].X > 1e-6
            }
            result["gamma_val"] = vars_map["gamma"].X

        return result

    @staticmethod
    def _get_linear_dual(constr):
        for attr_name in ("Pi", "BarPi"):
            try:
                return constr.getAttr(attr_name)
            except Exception:
                continue
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