import math
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

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
    from src.model.benders_model import IndexSets
    from src.model.benders_primal import PrimalSubproblem
    from src.model.benders_dual import DualSubproblem
else:
    from ..data_processor.loader import DataLoader
    from ..utils.output_processor import ResultProcessor
    from .. import config
    from ..data_processor.instance import Instance
    from .benders_model import IndexSets
    from .benders_primal import PrimalSubproblem
    from .benders_dual import DualSubproblem

XHat = Dict[str, int]
UHat = Dict[Tuple[str, int], int]
TOL = 1e-7

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
        self.time_limit = time_limit
        self.verbose = verbose

        self.model: Optional[gp.Model] = None
        self.x: Optional[gp.tupledict] = None
        self.u: Optional[gp.tupledict] = None
        self.theta: Optional[gp.Var] = None
        self.best_obj = -math.inf
        self.best_sol: Optional[Dict[str, Any]] = None
        self._visited_master_keys: set = set()

    def _build_master(self) -> None:
        data = self.data
        inst = self.inst

        mdl = gp.Model("BendersMaster_Primal")
        mdl.setParam("OutputFlag", 1 if self.verbose else 0)

        x = mdl.addVars(data.N + data.L_v, vtype=GRB.BINARY, name="x")
        u = mdl.addVars(data.L, data.S, vtype=GRB.BINARY, name="u")
        y = mdl.addVars(data.A_PD, lb=0.0, ub=1.0, name="y")

        fixed_cost = gp.quicksum(self.inst.f[u_idx] * x[u_idx] for u_idx in self.N)

        theta = mdl.addVar(lb=-GRB.INFINITY, ub=config.ORDER_REVENUE_COEF * data.total_demand, name="theta")
        mdl.setObjective(theta - fixed_cost, GRB.MAXIMIZE)

        # McCormick linearization for y[i,j] = x[i] * x[j] on binary x.
        for i, j in data.A_PD:
            mdl.addConstr(y[i, j] <= x[i], name=f"lin_y_ub_i_{i}_{j}")
            mdl.addConstr(y[i, j] <= x[j], name=f"lin_y_ub_j_{i}_{j}")
            mdl.addConstr(y[i, j] >= x[i] + x[j] - 1, name=f"lin_y_lb_{i}_{j}")

        # service level selection
        for l in data.L:
            mdl.addConstr(
                gp.quicksum(u[l, s] for s in data.S) == x[l],
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

        mw = data.service_wait_expr(u)
        # mdl.addConstr(
        #     mw <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[b] for b in data.L_hat),
        #     name="fleet_lb",
        # )
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

        # NOTE:
        # Legacy capacity pre-cuts tied to the old D-based LV-wait threshold are removed.
        # With the current u-based LV wait formulation, those cuts can over-restrict valid
        # master solutions and break primal/dual alignment.

        # 挂载变量到 model 对象，供 Callback 使用


        mdl._vars_x = x
        mdl._vars_u = u
        mdl._vars_y = y
        self.model = mdl
        self.x = x
        self.u = u
        self.theta = theta

    def get_solution_binary(self, model: gp.Model):
        x_hat = {
            n: 1 if model.cbGetSolution(self.x[n]) > 0.5 else 0
            for n in self.data.N + self.data.L_v
        }
        u_hat = {
            (l, s): 1 if model.cbGetSolution(self.u[l, s]) > 0.5 else 0
            for l in self.data.L
            for s in self.data.S
        }
        y_hat = {
            (i, j): float(model.cbGetSolution(model._vars_y[i, j]))
            for i, j in self.data.A_PD
        }
        tv = model.cbGetSolution(self.theta)
        return x_hat, u_hat, y_hat, tv

    def _master_solution_key(self, x_hat, u_hat):
        return (
            tuple(x_hat[n] for n in self.data.N + self.data.L_v),
            tuple(u_hat[l, s] for l in self.data.L for s in self.data.S),
        )

    def _build_cut_rhs(self, duals: dict, x_ref, u_ref, q_ref, use_gurobi_expr: bool):
        """Build cut RHS with the same objective builder used in DualSubproblem."""
        return DualSubproblem.build_objective_expr(
            data=self.data,
            inst=self.inst,
            x_ref=x_ref,
            u_ref=u_ref,
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
        """Evaluate affine cut RHS at (x_hat, u_hat) for consistency check."""
        duals = cd["duals"]
        data = self.data
        x_hat = cd["x_hat"]
        u_hat = cd["u_hat"]
        q_hat = {loc: data.service_wait_value(u_hat, [loc]) for loc in data.L}
        return self._build_cut_rhs(duals, x_hat, u_hat, q_hat, use_gurobi_expr=False)

    def _add_cut(self, model: gp.Model, cut_data: dict) -> None:
        """Add optimality cut using coefficients computed from subproblem dual multipliers."""
        duals = cut_data["duals"]
        data = self.data
        Q_var = {loc: data.service_wait_value(self.u, [loc]) for loc in data.L}
        expr = self._build_cut_rhs(duals, self.x, self.u, Q_var, use_gurobi_expr=True)

        model.cbLazy(self.theta <= expr)

    def _required_lv_flow(self, u_hat: UHat) -> float:
        if not self.data.L_v:
            return 0.0
        return sum(
            self.data.service_wait_value(u_hat, [lv]) / config.MAX_WAITING_FACTOR_LV
            for lv in self.data.L_v
        )

    def _flow_required_expr(self) -> gp.LinExpr:
        if not self.data.L_v:
            return gp.LinExpr(0.0)
        return (1.0 / config.MAX_WAITING_FACTOR_LV) * self.data.service_wait_expr(
            self.u, locations=self.data.L_v
        )

    @staticmethod
    def _edmonds_karp(
        nodes: List[str],
        edges: Dict[Tuple[str, str], float],
        source: str,
        sink: str,
    ) -> Tuple[float, Set[str]]:
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        residual: Dict[Tuple[str, str], float] = {}

        for (u_node, v_node), cap in edges.items():
            cap = max(0.0, float(cap))
            if cap <= TOL:
                continue
            if v_node not in adj[u_node]:
                adj[u_node].append(v_node)
            if u_node not in adj[v_node]:
                adj[v_node].append(u_node)
            residual[u_node, v_node] = residual.get((u_node, v_node), 0.0) + cap
            residual[v_node, u_node] = residual.get((v_node, u_node), 0.0)

        max_flow = 0.0
        while True:
            parent: Dict[str, Optional[str]] = {source: None}
            q = deque([source])
            while q and sink not in parent:
                cur = q.popleft()
                for nxt in adj[cur]:
                    if nxt in parent:
                        continue
                    if residual.get((cur, nxt), 0.0) > TOL:
                        parent[nxt] = cur
                        q.append(nxt)

            if sink not in parent:
                break

            path_cap = float("inf")
            v = sink
            while parent[v] is not None:
                u = parent[v]
                path_cap = min(path_cap, residual[u, v])
                v = u

            v = sink
            while parent[v] is not None:
                u = parent[v]
                residual[u, v] -= path_cap
                residual[v, u] = residual.get((v, u), 0.0) + path_cap
                v = u

            max_flow += path_cap

        reachable: Set[str] = set()
        q = deque([source])
        reachable.add(source)
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt in reachable:
                    continue
                if residual.get((cur, nxt), 0.0) > TOL:
                    reachable.add(nxt)
                    q.append(nxt)

        return max_flow, reachable

    def _compute_pd_mincut(
        self,
        x_hat: XHat,
        y_hat: Dict[Tuple[str, str], float],
    ) -> Tuple[float, Set[str], Set[str]]:
        source = "__src__"
        sink = "__snk__"
        nodes = [source, sink] + self.data.P + self.data.D
        edges: Dict[Tuple[str, str], float] = {}

        for i in self.data.P:
            edges[source, i] = self.data.H.get(i, 0.0) * x_hat[i]
        for i, j in self.data.A_PD:
            edges[i, j] = self.inst.h.get((i, j), 0.0) * max(0.0, y_hat.get((i, j), 0.0))
        for j in self.data.D:
            edges[j, sink] = self.data.H_hat.get(j, 0.0) * x_hat[j]

        max_flow, reachable = self._edmonds_karp(nodes, edges, source, sink)
        reachable_p = {i for i in self.data.P if i in reachable}
        reachable_d = {j for j in self.data.D if j in reachable}
        return max_flow, reachable_p, reachable_d

    def _add_flow_mincut_cut(
        self,
        model: gp.Model,
        reachable_p: Set[str],
        reachable_d: Set[str],
    ) -> None:
        lhs = gp.LinExpr()
        lhs += gp.quicksum(
            self.data.H.get(i, 0.0) * self.x[i]
            for i in self.data.P
            if i not in reachable_p
        )
        lhs += gp.quicksum(
            self.inst.h.get((i, j), 0.0) * model._vars_y[i, j]
            for i, j in self.data.A_PD
            if i in reachable_p and j not in reachable_d
        )
        lhs += gp.quicksum(
            self.data.H_hat.get(j, 0.0) * self.x[j]
            for j in reachable_d
        )
        model.cbLazy(lhs >= self._flow_required_expr())

    def _add_swap_energy_cuts(
        self,
        model: gp.Model,
        x_hat: XHat,
        u_hat: UHat,
    ) -> int:
        cuts_added = 0
        phi_key_set = set(self.data.phi_keys)
        for d in self.data.D:
            lv = self.inst.d_to_lv[d]
            if lv not in self.data.L_v:
                continue
            if x_hat.get(lv, 0) == 0:
                continue

            lv_wait = self.data.service_wait_value(u_hat, [lv])
            if lv_wait <= TOL:
                continue

            swap_opts = list(self.data.nearby_swaps.get(lv, []))
            base_opts = [
                b for b in self.data.L_hat
                if (d, b, "0") in phi_key_set
            ]
            rhs_nodes = sorted(set(swap_opts + base_opts))
            if not rhs_nodes:
                model.cbLazy(self.x[lv] == 0)
                cuts_added += 1
                continue

            if sum(x_hat.get(node, 0) for node in rhs_nodes) == 0:
                model.cbLazy(self.x[lv] <= gp.quicksum(self.x[node] for node in rhs_nodes))
                cuts_added += 1

        return cuts_added
    
    # alpha_2
    # epsilon_1
    # kappa_3


    def _add_nogood(self, model: gp.Model, cd: dict) -> None:
        """Feasibility cut: exclude the current binary combination."""
        xh = cd["x_hat"]
        ones = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 1]
        zeros = [self.x[n] for n in self.data.N + self.data.L_v if xh[n] == 0]
        model.cbLazy(
            gp.quicksum(1 - v for v in ones) + gp.quicksum(zeros) >= 1
        )

    def solve(self) -> Optional[float]:
        self._build_master()
        subPrimal = PrimalSubproblem(self.inst, self.data)
        subDual = DualSubproblem(self.inst, self.data)

        def callback(model: gp.Model, where: int):
            if where != GRB.Callback.MIPSOL:
                return
            xh, uh, yh, tv = self.get_solution_binary(model)
            master_key = self._master_solution_key(xh, uh)
            if master_key in self._visited_master_keys:
                return
            self._visited_master_keys.add(master_key)

            # Legacy min-cut pre-filters removed: they were derived for the previous
            # D-threshold LV wait model and are not valid under the current u-based form.

            # status, obj, cd = subPrimal.solve(xh, uh)
            status, obj, cd = subDual.solve(xh, uh)

            if obj is not None:
                rhs_at_inc = self._cut_check(cd)
                tol = 1e-5 * max(1.0, abs(obj))
                if abs(rhs_at_inc - obj) > tol:
                    status_primal, obj_primal, cd_primal = subPrimal.solve(xh, uh)
                    print(
                            "Inconsistent cut: "
                            f"RHS(x_hat,u_hat)={rhs_at_inc:.6f}, dual_obj={obj:.6f}, primal_obj={obj_primal:.6f}"
                        )
                    
            if obj is None:
                added = False
                req_flow = self._required_lv_flow(uh)
                if req_flow > TOL:
                    max_flow, reach_p, reach_d = self._compute_pd_mincut(xh, yh)
                    if max_flow + 1e-6 < req_flow:
                        if self.verbose:
                            print(
                                f"Flow infeasibility detected: required={req_flow:.6f}, maxflow={max_flow:.6f}. Adding min-cut lazy cut."
                            )
                        self._add_flow_mincut_cut(model, reach_p, reach_d)
                        added = True

                swap_cuts = self._add_swap_energy_cuts(model, xh, uh)
                if swap_cuts > 0:
                    if self.verbose:
                        print(f"Swap/energy structure infeasibility detected. Added {swap_cuts} cut(s).")
                    added = True

                if not added:
                    if self.verbose:
                        print("Infeasible subproblem with no structural certificate, adding nogood cut.")
                    _status, _obj, _cd = subPrimal.solve(xh, uh)
                    self._add_nogood(model, cd)
                return

            if cd["kind"] == "feasibility":
                self._add_nogood(model, cd)
                return
            
            if obj is not None:
                self._add_cut(model, cd)

            if obj is not None and obj > self.best_obj:
                self.best_obj = obj
                self.best_sol = {"x": xh, "u": uh, "obj": obj}

        self.model.setParam("LazyConstraints", 1)
        self.model.optimize(callback)
        self.model.write("benders_primal_master.lp")
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
        primal_solver = BendersMerge(inst, max_iter=50, verbose=True, time_limit=300)
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