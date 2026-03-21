import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
        theta = mdl.addVar(lb=0.0, ub=data.total_demand, name="theta")
        mdl.setObjective(theta, GRB.MAXIMIZE)

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

        fc = gp.quicksum(inst.f[n] * x[n] for n in data.N)
        mw = data.service_wait_expr(u)
        mdl.addConstr(fc + config.COST_DRONE * mw <= config.BUDGET, name="budget_lb")
        mdl.addConstr(
            mw <= config.MAX_FLEET_PER_BASE * gp.quicksum(x[b] for b in data.L_hat),
            name="fleet_lb",
        )
        mdl.addConstr(
            theta <= gp.quicksum(data.H[i] * x[i] for i in data.P),
            name="theta_P",
        )
        mdl.addConstr(
            theta <= gp.quicksum(data.H_hat[j] * x[j] for j in data.D),
            name="theta_D",
        )
        mdl.addConstr(
            theta <= gp.quicksum(inst.h[i, j] * y[i, j] for i, j in data.A_PD),
            name="theta_PD",
        )

        C_CONSTANT = 1.0 / config.MAX_WAITING_FACTOR_LV
        
        # 1.1 局部先验割 (Local Capacity Cut)
        for l in data.L_v:
            max_potential_demand = gp.quicksum(
                inst.h[i, j] * y[i, j]
                for i, j in data.A_PD 
                if (l, i) in data.A_LP  # 修正了原代码的 if 连用语法错误
            )
            mdl.addConstr(
                max_potential_demand >= C_CONSTANT * x[l],
                name=f"min_cap_threshold_{l}"
            )
            print(f"[Local Cut] Virtual base {l} with service requires at least {C_CONSTANT:.4f} capacity. Added cut.")

        # 1.2 全局先验割 (Global Capacity Cut) 
        # 全网开启的总容量，必须 >= 所有被激活的虚拟基地需要的下界总和
        global_potential_demand = gp.quicksum(
            inst.h[i, j] * y[i, j]
            for i, j in data.A_PD
        )
        mdl.addConstr(
            global_potential_demand >= C_CONSTANT * gp.quicksum(x[l] for l in data.L_v),
            name="global_min_cap_threshold"
        )
        print(f"[Global Cut] Total potential demand from all active edges must cover the sum of {C_CONSTANT:.4f} times the number of active virtual bases. Added cut.")

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
        tv = model.cbGetSolution(self.theta)
        return x_hat, u_hat, tv

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
            kappa2=duals["kappa2"],
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
            xh, uh, tv = self.get_solution_binary(model)
            master_key = self._master_solution_key(xh, uh)
            if master_key in self._visited_master_keys:
                return
            self._visited_master_keys.add(master_key)

            C_CONSTANT = 1.0 / config.MAX_WAITING_FACTOR_LV
            TOL = 1e-5
            
            # --- 提前提取当前解的活跃状态，减少后续循环计算 ---
            active_v_bases = [l for l in self.data.L_v if xh[l] > 0.5]
            active_edges = [(i, j) for (i, j) in self.data.A_PD if xh[i] > 0.5 and xh[j] > 0.5]

            # 1. 快速的单点局部检查 (Local Cut-Set Filter)
            local_fail = False
            for l in active_v_bases:
                available_demand = sum(
                    self.inst.h[i, j] 
                    for (i, j) in active_edges 
                    if (l, i) in self.data.A_LP
                )
                if available_demand < C_CONSTANT - TOL:
                    print(f"[Local Cut] Virtual base {l} lacks capacity: {available_demand:.4f} < {C_CONSTANT:.4f}. Rejected.")
                    max_demand_expr = gp.quicksum(
                        self.inst.h[i, j] * model._vars_y[i, j]
                        for (i, j) in self.data.A_PD
                        if (l, i) in self.data.A_LP
                    )
                    model.cbLazy(max_demand_expr >= C_CONSTANT * model._vars_x[l])
                    local_fail = True
            
            if local_fail:
                return

            # 2. 基于最大流的精确子集检查 (Max-Flow Min-Cut Subset Filter)
            # 这能发现任意组合 (如 {v2, v3}) 共享容量不足的情况，绝不会漏 Cut
            if len(active_v_bases) > 1:
                import networkx as nx
                G = nx.DiGraph()
                G.add_node("S")
                G.add_node("T")
                
                # (1) 源点 -> 虚拟基站 (该边容量为基站的最低需求)
                for l in active_v_bases:
                    G.add_edge("S", l, capacity=C_CONSTANT)
                    
                # (2) 配送边 -> 汇点 (该边容量为该配送能提供的最大订单流)
                for (i, j) in active_edges:
                    G.add_edge((i, j), "T", capacity=self.inst.h[i, j])
                    
                # (3) 虚拟基站 -> 配送边 (连通性映射，容量正无穷)
                for l in active_v_bases:
                    for (i, j) in active_edges:
                        if (l, i) in self.data.A_LP:
                            G.add_edge(l, (i, j), capacity=float('inf'))
                            
                # 计算最大流和最小割
                flow_value, partition = nx.minimum_cut(G, "S", "T")
                required_demand = C_CONSTANT * len(active_v_bases)
                
                # 如果最大流小于总需求，说明至少存在一个子集发生拥堵（无解）
                if flow_value < required_demand - TOL:
                    # partition[0] 是包含源点 S 的集合，里面包括了所有"被饿死"(缺乏边覆盖)的基站
                    S_set = partition[0] 
                    starved_bases = [node for node in S_set if node in active_v_bases]
                    
                    if starved_bases:
                        # 找出这些困局基站所能触达的**所有候选边**（用于写 Lazy Cut）
                        reachable_edges = set()
                        for l in starved_bases:
                            for i, j in self.data.A_PD:
                                if (l, i) in self.data.A_LP:
                                    reachable_edges.add((i, j))
                        
                        shared_available_demand = sum(self.inst.h[i, j] for (i, j) in reachable_edges if (i, j) in active_edges)
                        starved_required = C_CONSTANT * len(starved_bases)
                        
                        print(f"[Min-Cut Filter] Active bases {starved_bases} share only {shared_available_demand:.4f} capacity, need {starved_required:.4f}. Rejected.")
                        
                        # 构造子集 Cut
                        shared_demand_expr = gp.quicksum(
                            self.inst.h[i, j] * model._vars_y[i, j]
                            for (i, j) in reachable_edges
                        )
                        required_expr = gp.quicksum(
                            C_CONSTANT * model._vars_x[l]
                            for l in starved_bases
                        )
                        model.cbLazy(shared_demand_expr >= required_expr)


                        # model.cbLazy(gp.quicksum(self.inst.h[i, j] * model._vars_x[i] * model._vars_x[j] for (i, j) in reachable_edges) >= C_CONSTANT * gp.quicksum(model._vars_x[l] for l in starved_bases))
                        return 

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
                print("Infeasible subproblem, adding nogood cut.")
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