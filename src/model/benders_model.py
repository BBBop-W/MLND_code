"""
Benders Decomposition for the MLND Model.

Implements a Generalised Benders Decomposition where:

  * **Master problem** contains all integer and continuous variables of the
    original model *except* the SOCP queuing constraint, which is replaced by
    outer-approximation (OA) tangent cuts that are tightened iteratively.
  * **Sub-problem** evaluates the exact queuing function xi >= lam/(mu - lam)
    at the master's optimal lambda* values and generates new OA tangent cuts.

This is equivalent to an Outer-Approximation (OA) algorithm for Mixed-Integer
SOCP, a well-known specialisation of Generalised Benders Decomposition for
convex nonlinear constraints.

Algorithm
---------
1. Build master MIP = original model with SOCP replaced by a few OA cuts.
2. Solve master to optimality  -> get (x*, u*, gamma*, lam*).
3. For each node i, evaluate xi_exact = lam*_i / (mu_i - lam*_i).
   If xi* < xi_exact (OA too loose), add a new tangent cut at lam*_i.
4. Repeat until no violated OA cut is found (gap <= tol) or iteration limit.
"""

import math
import time

import gurobipy as gp
from gurobipy import GRB

from ..data.instance import Instance
from .. import config


class BendersDecomposition:
    """OA-based Benders decomposition for MLND MISOCP."""

    def __init__(self, instance: Instance,
                 max_iter=50, gap_tol=1e-4, time_limit=3600,
                 mip_tl=300, verbose=True):
        self.inst = instance
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        self.time_limit = time_limit
        self.mip_tl = mip_tl
        self.verbose = verbose

        self.best_obj = None
        self.history = []

    # ------------------------------------------------------------------
    def _build_master(self):
        """Build the full MILP master: original model with SOCP -> OA."""
        inst = self.inst

        L_hat = inst.L_hat
        L_v = inst.L_v if config.USE_VIRTUAL_BASES_OPT else []
        L = L_hat + L_v
        P, D, E, N = inst.P, inst.D, inst.E, inst.N

        ok = (lambda a: True) if config.USE_VIRTUAL_BASES_OPT else (
            lambda a: not a[0].startswith('V_') and not a[1].startswith('V_'))
        A_all = [a for a in inst.A if ok(a)]
        A_PD = [a for a in inst.A_PD if a in set(A_all)]
        A_LP = [a for a in inst.A_LP if a in set(A_all)]
        A_DL = [a for a in inst.A_DL if a in set(A_all)]
        A = A_PD + A_LP + A_DL

        E_uv = {a: inst.E_uv[a] for a in A if a in inst.E_uv}
        S = list(range(len(config.SERVICE_LEVELS)))
        rho = config.SERVICE_LEVELS
        D_set = set(D)
        L_hat_set = set(L_hat)

        m = gp.Model("Benders_Master")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', self.mip_tl)

        # -- Variables --
        x_nodes = N + L_v
        x = m.addVars(x_nodes, vtype=GRB.BINARY, name="x")
        u = m.addVars(L, S, vtype=GRB.BINARY, name="u")
        y = m.addVars(A_LP, lb=0.0, ub=1.0, name="y")
        w_keys = [(l, i, j) for l, i in A_LP for j in D if (i, j) in set(A_PD)]
        w = m.addVars(w_keys, lb=0.0, name="w")
        D_var = m.addVars(A_LP, lb=0.0, name="D")
        z_keys = [(l, i, s) for l, i in A_LP for s in S]
        z = m.addVars(z_keys, lb=0.0, name="z")
        psi = m.addVars(A, lb=0.0, name="psi")

        phi_keys = []
        for u_n, v_n in A:
            d_uv = inst.get_dist(u_n, v_n)
            if d_uv <= config.MAX_DIRECT_DISTANCE or (
                    u_n in D_set and v_n.startswith('V_')):
                phi_keys.append((u_n, v_n, '0'))
            for k in E_uv.get((u_n, v_n), []):
                phi_keys.append((u_n, v_n, k))
        phi_keys_set = set(phi_keys)

        phi = {}
        for u_n, v_n, k in phi_keys:
            phi[u_n, v_n, k] = m.addVar(lb=0.0, name=f"phi_{u_n}_{v_n}_{k}")

        xi = m.addVars(P + D, lb=0.0, name="xi")
        lam = m.addVars(P + D, lb=0.0, name="lam")
        gamma = m.addVar(lb=0.0, name="gamma")

        # -- Objective --
        m.setObjective(gp.quicksum(psi[i, j] for i, j in A_PD), GRB.MAXIMIZE)

        # -- Constraints --
        # 1. Assignment & service level
        A_LP_set = set(A_LP)
        for i in P:
            ls = [l for l in L if (l, i) in A_LP_set]
            if ls:
                m.addConstr(gp.quicksum(y[l, i] for l in ls) <= x[i])
        for l, i in A_LP:
            m.addConstr(y[l, i] <= x[l])
        for l in L:
            m.addConstr(gp.quicksum(u[l, s] for s in S) == x[l])

        if config.USE_VIRTUAL_BASES_OPT:
            for d in D:
                m.addConstr(x[inst.d_to_lv[d]] <= x[d])

        # 2. PWLA linearization
        A_PD_set = set(A_PD)
        H = {}
        for i in P:
            H[i] = sum(inst.h.get((i, j), 0) for j in D if (i, j) in A_PD_set)

        for l, i in A_LP:
            Hi = H.get(i, 0)
            for j in D:
                if (i, j) not in A_PD_set:
                    continue
                m.addConstr(w[l, i, j] <= y[l, i])
                m.addConstr(w[l, i, j] <= x[j])
                m.addConstr(w[l, i, j] >= y[l, i] - (1 - x[j]))
            m.addConstr(D_var[l, i] == gp.quicksum(
                inst.h.get((i, j), 0) * w[l, i, j]
                for j in D if (i, j) in A_PD_set))
            for s in S:
                m.addConstr(z[l, i, s] <= D_var[l, i])
                m.addConstr(z[l, i, s] <= Hi * u[l, s])
                m.addConstr(z[l, i, s] >= D_var[l, i] - Hi * (1 - u[l, s]))
            m.addConstr(psi[l, i] == gp.quicksum(
                rho[s] * z[l, i, s] for s in S))

        # 3. Flow conservation & decomposition
        out = {}
        inn = {}
        for u_n, v_n in A:
            out.setdefault(u_n, []).append((u_n, v_n))
            inn.setdefault(v_n, []).append((u_n, v_n))

        for v_n in N + L_v:
            m.addConstr(
                gp.quicksum(psi[a] for a in inn.get(v_n, [])) ==
                gp.quicksum(psi[a] for a in out.get(v_n, [])))

        for i, j in A_PD:
            m.addConstr(psi[i, j] <= inst.h[i, j])

        phi_for_arc = {}
        for u_n, v_n, k in phi_keys:
            phi_for_arc.setdefault((u_n, v_n), []).append(k)

        for u_n, v_n in A:
            ks = phi_for_arc.get((u_n, v_n), [])
            m.addConstr(psi[u_n, v_n] == gp.quicksum(
                phi[u_n, v_n, k] for k in ks))

        # 4. QoS
        for nd in P + D:
            m.addConstr(
                gp.quicksum(psi[a] for a in out.get(nd, [])) <=
                config.MAX_WAITING_FACTOR_P * x[nd])

        phi_thru = {}
        for u_n, v_n, k in phi_keys:
            if k != '0':
                phi_thru.setdefault(k, []).append((u_n, v_n, k))

        for k in E:
            m.addConstr(
                gp.quicksum(phi[key] for key in phi_thru.get(k, [])) <=
                config.MAX_FLOW_E * x[k])

        if config.USE_VIRTUAL_BASES_OPT:
            for lv in L_v:
                rr_expr = gp.quicksum(
                    (rho[s] / (1 - rho[s])) * u[lv, s]
                    for s in S if rho[s] < 1.0)
                lp_arcs = [(u_n, i_n) for u_n, i_n in A_LP if u_n == lv]
                m.addConstr(rr_expr <= config.MAX_WAITING_FACTOR_LV *
                            gp.quicksum(psi[a] for a in lp_arcs))

        # 5. Lambda definition
        for nd in P + D:
            m.addConstr(lam[nd] == gp.quicksum(
                psi[a] for a in out.get(nd, [])))
            m.addConstr(lam[nd] <= inst.mu[nd] - 1e-3)

        # 6. Initial OA tangent cuts for xi >= lam/(mu-lam)
        oa_pts = {}
        for nd in P + D:
            mu_i = inst.mu[nd]
            oa_pts[nd] = []
            for frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                lam0 = mu_i * frac
                oa_pts[nd].append(lam0)
                den = mu_i - lam0
                f0 = lam0 / den
                fp = mu_i / (den * den)
                m.addConstr(xi[nd] >= (f0 - fp * lam0) + fp * lam[nd])

        # 7. Energy balance
        if config.USE_VIRTUAL_BASES_OPT:
            cap_eff = ((1 - config.ENERGY_SAFETY_MARGIN_EPSILON) *
                       config.BATTERY_CAPACITY_OMEGA_UPPER)
            for lv in L_v:
                wait = gp.quicksum(
                    (rho[s] / (1 - rho[s])) * u[lv, s]
                    for s in S if rho[s] < 1.0)
                transit = gp.LinExpr()
                replen = gp.LinExpr()
                for u_n, i_n in A_LP:
                    if u_n != lv:
                        continue
                    if (u_n, i_n, '0') in phi_keys_set:
                        transit += phi[u_n, i_n, '0'] * inst.get_time(u_n, i_n)
                    for k in E_uv.get((u_n, i_n), []):
                        if (u_n, i_n, k) in phi_keys_set:
                            td = inst.get_time(u_n, k) + inst.get_time(k, i_n)
                            transit += phi[u_n, i_n, k] * td
                            replen += phi[u_n, i_n, k]
                m.addConstr(config.ENERGY_RATE_OMEGA * (wait + transit) <=
                            cap_eff * replen)

        cap_eff = ((1 - config.ENERGY_SAFETY_MARGIN_EPSILON) *
                   config.BATTERY_CAPACITY_OMEGA_UPPER)
        for i_n in P:
            transit = gp.LinExpr()
            replen = gp.LinExpr()
            for u_n, v_n in A_PD:
                if u_n != i_n:
                    continue
                if (u_n, v_n, '0') in phi_keys_set:
                    transit += phi[u_n, v_n, '0'] * inst.get_time(u_n, v_n)
                for k in E_uv.get((u_n, v_n), []):
                    if (u_n, v_n, k) in phi_keys_set:
                        td = inst.get_time(u_n, k) + inst.get_time(k, v_n)
                        transit += phi[u_n, v_n, k] * td
                        replen += phi[u_n, v_n, k]
            m.addConstr(config.ENERGY_RATE_OMEGA * (xi[i_n] + transit) <=
                        cap_eff * replen)

        for j_n in D:
            transit = gp.LinExpr()
            replen = gp.LinExpr()
            for u_n, v_n in out.get(j_n, []):
                if (u_n, v_n, '0') in phi_keys_set:
                    transit += phi[u_n, v_n, '0'] * inst.get_time(u_n, v_n)
                    if v_n in L_hat_set:
                        replen += phi[u_n, v_n, '0']
                for k in E_uv.get((u_n, v_n), []):
                    if (u_n, v_n, k) in phi_keys_set:
                        td = inst.get_time(u_n, k) + inst.get_time(k, v_n)
                        transit += phi[u_n, v_n, k] * td
                        replen += phi[u_n, v_n, k]
            m.addConstr(config.ENERGY_RATE_OMEGA * (xi[j_n] + transit) <=
                        cap_eff * replen)

        # 8. Fleet & budget
        fleet = gp.LinExpr()
        fleet += gp.quicksum(
            (rho[s] / (1 - rho[s])) * u[l, s]
            for l in L for s in S if rho[s] < 1.0)
        fleet += gp.quicksum(xi[nd] for nd in P + D)
        for u_n, v_n in A:
            if (u_n, v_n, '0') in phi_keys_set:
                fleet += phi[u_n, v_n, '0'] * inst.get_time(u_n, v_n)
            for k in E_uv.get((u_n, v_n), []):
                if (u_n, v_n, k) in phi_keys_set:
                    td = inst.get_time(u_n, k) + inst.get_time(k, v_n)
                    fleet += phi[u_n, v_n, k] * td
        m.addConstr(fleet <= gamma)
        m.addConstr(gp.quicksum(inst.f[n] * x[n] for n in N) +
                    config.COST_DRONE * gamma <= config.BUDGET)
        m.addConstr(gamma <= config.MAX_FLEET_PER_BASE *
                    gp.quicksum(x[l] for l in L_hat))

        m.update()

        # Store for later
        self._m = m
        self._xi = xi
        self._lam = lam
        self._oa_pts = oa_pts
        self._P = P
        self._D = D
        self._A_PD = A_PD
        self._psi = psi

    # ------------------------------------------------------------------
    def _add_oa_cuts(self):
        """Evaluate exact queuing at lam*, add tangent if OA is violated.
        Returns max violation."""
        max_viol = 0.0
        cuts_added = 0
        for nd in self._P + self._D:
            lam_val = self._lam[nd].X
            xi_val = self._xi[nd].X
            mu_i = self.inst.mu[nd]

            if lam_val < 1e-8:
                continue
            if lam_val >= mu_i - 1e-6:
                continue

            xi_exact = lam_val / (mu_i - lam_val)
            viol = xi_exact - xi_val

            if viol > 1e-6:
                max_viol = max(max_viol, viol)
                # Check if we already have a nearby OA point
                existing = self._oa_pts[nd]
                if any(abs(lam_val - e) < 0.5 for e in existing):
                    continue
                existing.append(lam_val)
                den = mu_i - lam_val
                f0 = lam_val / den
                fp = mu_i / (den * den)
                self._m.addConstr(
                    self._xi[nd] >= (f0 - fp * lam_val) + fp * self._lam[nd])
                cuts_added += 1

        return max_viol, cuts_added

    # ------------------------------------------------------------------
    def solve(self):
        t0 = time.time()
        if self.verbose:
            print("=" * 72)
            print("BENDERS (OA) DECOMPOSITION FOR MLND MODEL")
            print("=" * 72)

        if self.verbose:
            print("Building master (full MIP + OA)...")
        self._build_master()
        if self.verbose:
            nv = self._m.NumVars
            nc = self._m.NumConstrs
            print(f"Master: {nv} vars, {nc} constrs")

        prev_obj = None

        for it in range(1, self.max_iter + 1):
            elapsed = time.time() - t0
            if elapsed > self.time_limit:
                if self.verbose:
                    print(f"\nTime limit ({self.time_limit}s)")
                break

            remaining = self.time_limit - elapsed
            self._m.setParam('TimeLimit', min(self.mip_tl, remaining))
            self._m.optimize()

            if self._m.SolCount == 0:
                if self.verbose:
                    print(f"Iter {it}: No solution found (status={self._m.Status})")
                break

            obj = self._m.ObjVal
            gap = self._m.MIPGap if hasattr(self._m, 'MIPGap') else 0

            max_viol, cuts_added = self._add_oa_cuts()

            if self.verbose:
                print(f"Iter {it:3d} | Obj={obj:10.4f} | "
                      f"MIPGap={gap:8.4%} | "
                      f"MaxViol={max_viol:10.6f} | "
                      f"NewCuts={cuts_added:3d} | "
                      f"{elapsed:.1f}s")

            self.best_obj = obj
            self.history.append({
                'iter': it, 'obj': obj, 'gap': gap,
                'max_viol': max_viol, 'cuts': cuts_added,
                'time': elapsed
            })

            if cuts_added == 0 and max_viol <= self.gap_tol:
                if self.verbose:
                    print(f"\nConverged! max_viol={max_viol:.8f}")
                break

            if prev_obj is not None and abs(obj - prev_obj) < 1e-6 and cuts_added == 0:
                if self.verbose:
                    print("\nNo progress and no new cuts — converged.")
                break
            prev_obj = obj

        elapsed = time.time() - t0
        if self.verbose:
            print("=" * 72)
            print(f"Done  best={self.best_obj:.4f}  time={elapsed:.1f}s")
            print("=" * 72)
        return self.best_obj

    def get_solution(self):
        """Extract solution details."""
        if self._m.SolCount == 0:
            return None

        sol = {'objective': self.best_obj, 'history': self.history}

        # Extract x, u, gamma
        sol['x'] = {}
        for v in self._m.getVars():
            if v.VarName.startswith('x['):
                nd = v.VarName[2:-1]
                sol['x'][nd] = v.X
        sol['u'] = {}
        for v in self._m.getVars():
            if v.VarName.startswith('u['):
                parts = v.VarName[2:-1].split(',')
                sol['u'][(parts[0], int(parts[1]))] = v.X

        gamma_v = self._m.getVarByName('gamma')
        sol['gamma'] = gamma_v.X if gamma_v else 0

        return sol
