"""Demand, fleet, cost and gap figures across saved instances."""

from __future__ import annotations

import numpy as np
from .style import plt, _save, _grid

COLORS = dict(
    pickup="#176b87",
    home="#63a8b5",
    direct="#ccd1d8",
    idle="#b8bdc6",
    first="#69aab9",
    loaded="#176b87",
    returning="#96cad2",
    wait="#d09335",
    swap="#ae729d",
    handling="#d8b7ca",
    empty="#dd704a",
    fixed="#aa79a8",
    fleet="#d39a43",
    flying="#5b8e9c",
    rider="#a4b9ce",
)
FLEET_COMPONENTS = [
    ("基地等待", "idle", ["idle"]),
    ("接餐飞行", "first", ["pre_pickup_flight"]),
    ("带餐飞行", "loaded", ["loaded_flight"]),
    ("任务返回", "returning", ["return_flight"]),
    ("空中等餐", "wait", ["sync_wait"]),
    ("任务换电", "swap", ["pre_pickup_swap", "loaded_swap", "return_swap"]),
    ("取放餐", "handling", ["pickup_handling", "delivery_handling"]),
    ("空载重调度", "empty", ["empty_flight", "empty_swap"]),
]
COST_COMPONENTS = [
    ("纯骑手", "direct", ["direct_rider"]),
    ("首末程骑手", "rider", ["first_rider", "terminal_rider"]),
    ("飞行与悬停", "flying", ["service_flight", "aerial_wait", "empty_flight"]),
    ("机队", "fleet", ["fleet"]),
    ("设施及航线", "fixed", ["base_fixed", "locker_fixed", "swap_fixed", "link_fixed"]),
]


def _scale_charts(cases, out):
    labels = [f'H{c["row"]["H"]:03d}\n{c["row"]["OD"]:,} OD' for c in cases]
    y = np.arange(len(cases))
    height = max(4.0, len(cases) * 0.85 + 1.6)
    fig, ax = plt.subplots(figsize=(8.0, height))
    left = np.zeros(len(cases))
    for key, label in [
        ("pickup", "无人机＋自提"),
        ("home", "无人机＋骑手到家"),
        ("direct", "全程纯骑手"),
    ]:
        vals = np.array([c["mode"][key] for c in cases])
        ax.barh(y, vals, left=left, color=COLORS[key], label=label, height=0.58)
        for yy, v, x in zip(y, vals, left):
            if v > 0.065 * max(c["row"]["demand"] for c in cases):
                ax.text(
                    x + v / 2,
                    yy,
                    f"{v:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if key == "pickup" else "#203340",
                )
        left += vals
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("订单 / 小时（各部分合计为全部潜在需求）")
    ax.set_title("各规模订单服务方式", loc="left", pad=42)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
    _grid(ax)
    _save(fig, out, "01_orders")
    for name, title, components, table, unit in [
        ("02_fleet", "平均机队占用", FLEET_COMPONENTS, "fleet", "无人机数量"),
        ("03_costs", "总成本构成", COST_COMPONENTS, "costs", "元 / 小时"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, height + 0.35))
        left = np.zeros(len(cases))
        for label, color, keys in components:
            vals = np.array([sum(c[table].get(k, 0.0) for k in keys) for c in cases])
            ax.barh(y, vals, left=left, color=COLORS[color], label=label, height=0.58)
            left += vals
        for yy, v in zip(y, left):
            ax.text(v, yy, f"  {v:,.1f}", va="center", fontsize=11)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_xlim(0, max(left) * 1.12 if max(left) > 0 else 1)
        ax.set_title(title, loc="left", pad=65)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(-0.01, 1.01),
            ncol=4 if table == "fleet" else 3,
            frameon=False,
        )
        _grid(ax)
        _save(fig, out, name)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.3), layout="constrained")
    names = [f'H{c["row"]["H"]:03d}' for c in cases]
    x = np.arange(len(cases))
    vals = [c["row"]["cost_per_order"] for c in cases]
    base = [
        (
            c["row"]["baseline_cost"] / c["row"]["demand"]
            if c["row"]["baseline_cost"]
            else np.nan
        )
        for c in cases
    ]
    axes[0].bar(x - 0.17, base, 0.32, color=COLORS["direct"], label="全程骑手基准")
    axes[0].bar(x + 0.17, vals, 0.32, color=COLORS["pickup"], label="当前方案")
    for xx, v in zip(x, vals):
        axes[0].text(xx + 0.17, v, f"{v:.2f}", ha="center", va="bottom")
    axes[0].set_ylabel("元 / 潜在订单")
    axes[0].set_xticks(x, names)
    axes[0].legend(frameon=False)
    axes[0].set_title("平均成本", loc="left")
    _grid(axes[0], "y")
    gaps = [c["row"]["gap_pct"] or 0.0 for c in cases]
    axes[1].bar(x, gaps, 0.55, color=COLORS["fleet"])
    for xx, v, c in zip(x, gaps, cases):
        axes[1].text(
            xx,
            v,
            "—" if c["row"]["gap_pct"] is None else f"{v:.2f}%",
            ha="center",
            va="bottom",
        )
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("最优性 Gap（%）")
    axes[1].set_title("求解结束时的上下界差距", loc="left")
    _grid(axes[1], "y")
    _save(fig, out, "04_cost_and_gap")
