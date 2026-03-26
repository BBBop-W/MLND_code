from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src import config

XHat = Dict[str, int]
EtaHat = Dict[Tuple[str, int], int]


@dataclass
class FlowCut:
    lhs_lvs: List[str]
    rhs_x_coeff: Dict[str, float]
    rhs_xx_coeff: Dict[Tuple[str, str], float]
    rhs_const: float
    cut_value: float
    required: float


class ContinuityFlowSeparator:
    """Separate continuity-flow feasibility cuts using max-flow/min-cut on a single-commodity graph."""

    def __init__(self, data: Any):
        self.data = data
        self.src = "__SRC__"
        self.sink = "__SNK__"
        self.big_m = max(1.0, data.total_demand) * 10.0  # 稍微放大 Big-M 确保无穷大边绝不会被切断
        self.sep_eps = 1e-6

    def demand_coeff(self, lv: str, s: int) -> float:
        # Required outflow at lv is q_lv / MAX_WAITING_FACTOR_LV.
        return self.data.service_wait_t.get(s, 0.0) / config.MAX_WAITING_FACTOR_LV

    def demand_hat(self, eta_hat: EtaHat, lv: str) -> float:
        return sum(self.demand_coeff(lv, s) * eta_hat.get((lv, s), 0) for s in self.data.S)

    def _build_graph(self, x_hat: XHat, eta_hat: EtaHat) -> Tuple[nx.DiGraph, Dict[str, float], float]:
        if not self.data.L_v:
            return nx.DiGraph(), {}, 0.0

        demand_lv = {lv: self.demand_hat(eta_hat, lv) for lv in self.data.L_v}
        required = sum(demand_lv.values())

        g = nx.DiGraph()
        g.add_node(self.src)
        g.add_node(self.sink)

        # ---------------------------------------------------------
        # 修复 1: 拆分 L_v 节点 (实现 Hoffman 无源无汇循环流转化)
        # ---------------------------------------------------------
        for lv in self.data.L_v:
            lvin = f"{lv}__in"
            lvout = f"{lv}__out"
            dem = max(0.0, demand_lv[lv])
            
            # S -> lv__out: 强制把要求的发货量推入网络
            g.add_edge(self.src, lvout, capacity=dem, kind="src_lv", lv=lv)
            # lv__in -> T: 强制要求网络把货运回来结算
            g.add_edge(lvin, self.sink, capacity=dem, kind="lv_sink", lv=lv)
            # lv__in -> lv__out: 允许网络超额循环（多余的流可以直接穿过去，不占用结算容量）
            g.add_edge(lvin, lvout, capacity=self.big_m, kind="lv_internal", lv=lv)

        # P 节点拆分
        for p in self.data.P:
            pin = f"{p}__in"
            pout = f"{p}__out"
            cap = max(0.0, config.MAX_WAITING_FACTOR_P * float(x_hat.get(p, 0.0)))
            g.add_edge(pin, pout, capacity=cap, kind="p_node", p=p)

        # D 节点拆分
        for d in self.data.D:
            din = f"{d}__in"
            dout = f"{d}__out"
            cap = max(0.0, config.MAX_WAITING_FACTOR_D * float(x_hat.get(d, 0.0)))
            g.add_edge(din, dout, capacity=cap, kind="d_node", d=d)

        # ---------------------------------------------------------
        # 修复 2: 拆分物理基站 L (让 NetworkX 能够识别节点容量)
        # ---------------------------------------------------------
        for l in self.data.L:
            lin = f"{l}__in"
            lout = f"{l}__out"
            cap = self.big_m * float(x_hat.get(l, 0.0))
            g.add_edge(lin, lout, capacity=cap, kind="l_node", l=l)

        # L_v -> P (虚拟基站发货去 P)
        for lv, p in self.data.A_LP:
            if lv in self.data.L_v:
                g.add_edge(f"{lv}__out", f"{p}__in", capacity=self.big_m, kind="lv_p", lv=lv, p=p)


        # P -> D (物理取送货)
        for p, d in self.data.A_PD:
            base_cap = max(0.0, self.data.inst.h.get((p, d), 0.0))
            cap = base_cap * float(x_hat.get(p, 0.0)) * float(x_hat.get(d, 0.0))
            g.add_edge(
                f"{p}__out",
                f"{d}__in",
                capacity=cap,
                base_cap=base_cap,
                kind="pd",
                p=p,
                d=d,
            )

        # L -> P (物理基站发货去 P)
        for l, p in self.data.A_LP:
            if l in self.data.L:
                g.add_edge(f"{l}__out", f"{p}__in", capacity=self.big_m, kind="l_p", l=l, p=p)

        # D -> L (流回物理基站)
        for d, l in self.data.A:
            if d in self.data.D and l in self.data.L:
                g.add_edge(f"{d}__out", f"{l}__in", capacity=self.big_m, kind="d_l", d=d, l=l)

        # D -> L_v (流回虚拟基站，完成循环闭环)
        for u, v in self.data.A:
            if u in self.data.D and v in self.data.L_v:
                g.add_edge(f"{u}__out", f"{v}__in", capacity=self.big_m, kind="d_lv", d=u, lv=v)

        return g, demand_lv, required


    def separate(self, x_hat: XHat, eta_hat: EtaHat) -> Optional[FlowCut]:
        g, demand_lv, required = self._build_graph(x_hat, eta_hat)
        if required <= self.sep_eps:
            return None

        cut_value, (s_set, t_set) = nx.minimum_cut(g, self.src, self.sink, capacity="capacity")
        
        # 如果最大流(最小割) >= 必须流动的总量，说明当前网络完全可行，不产生 Cut
        if cut_value >= required - self.sep_eps:
            return None

        # ---------------------------------------------------------
        # 精确提取被憋住的虚拟源 (被切断的循环流)
        # 根据 Hoffman 定理: 只有出点在 S 集合且入点在 T 集合的 lv, 才是受害者
        # ---------------------------------------------------------
        lhs_lvs = [
            lv for lv in self.data.L_v 
            if f"{lv}__out" in s_set and f"{lv}__in" not in s_set and demand_lv[lv] > self.sep_eps
        ]
        
        # 安全回退机制: 如果某种极端割集导致上述判断失效，提取所有有需求的节点
        if not lhs_lvs:
            lhs_lvs = [lv for lv in self.data.L_v if demand_lv[lv] > self.sep_eps]

        rhs_x_coeff: Dict[str, float] = {}
        rhs_xx_coeff: Dict[Tuple[str, str], float] = {}
        rhs_const = 0.0

        for u in s_set:
            for v in g.successors(u):
                if v not in t_set:
                    continue
                edge = g[u][v]
                kind = edge.get("kind")
                cap = float(edge.get("capacity", 0.0))

                # 这些边属于虚拟源头/结算体系，其容量已经在 lhs_lvs 中体现了，不能加到右侧约束里
                if kind in ["src_lv", "lv_sink", "lv_internal"]:
                    continue

                # 提取瓶颈节点/边的系数
                if kind == "p_node":
                    p = edge["p"]
                    rhs_x_coeff[p] = rhs_x_coeff.get(p, 0.0) + config.MAX_WAITING_FACTOR_P
                elif kind == "d_node":
                    d = edge["d"]
                    rhs_x_coeff[d] = rhs_x_coeff.get(d, 0.0) + config.MAX_WAITING_FACTOR_D
                elif kind == "l_node":
                    # ---------------------------------------------------------
                    # 修复 3: 提取物理基站 L 的系数，防止将其丢进常数项
                    # ---------------------------------------------------------
                    l = edge["l"]
                    rhs_x_coeff[l] = rhs_x_coeff.get(l, 0.0) + self.big_m
                elif kind == "pd":
                    p = edge["p"]
                    d = edge["d"]
                    base_cap = float(edge.get("base_cap", 0.0))
                    rhs_xx_coeff[p, d] = rhs_xx_coeff.get((p, d), 0.0) + base_cap
                else:
                    # 兜底逻辑：如果切到了某种容量为 big_m 的物理连通边，直接加到常数项里
                    # (这通常意味着网络连通性彻底断裂，加上一个巨大的常数会让这个割变得极强)
                    rhs_const += cap

        return FlowCut(
            lhs_lvs=lhs_lvs,
            rhs_x_coeff=rhs_x_coeff,
            rhs_xx_coeff=rhs_xx_coeff,
            rhs_const=rhs_const,
            cut_value=float(cut_value),
            required=float(required),
        )