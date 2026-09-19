"""Single-case operating and convergence figures."""

from __future__ import annotations

import numpy as np
from .style import plt, _style, _save, _grid
from .formatting import _tr


def figure(case, metrics, out, lang):
    _style()
    if lang == "en":
        plt.rcParams["font.family"] = "DejaVu Sans"
    T = lambda en, cn: _tr(en, cn, lang)
    f = case["fleet"]
    c = case["costs"]
    d = case["mode"]
    H = case["row"]["H"]
    datasets = [
        (
            T("Order fulfillment", "订单服务方式"),
            T(
                ["Drone + self-pickup", "Drone + home courier", "Courier only"],
                ["无人机＋自提", "无人机＋末端骑手", "全程骑手"],
            ),
            [d["pickup"], d["home"], d["direct"]],
            T("Orders per hour", "单/小时"),
        ),
        (
            T("Average fleet occupancy", "平均机队占用"),
            T(
                [
                    "At bases",
                    "Flight to pickup",
                    "Loaded flight",
                    "Mission return",
                    "Service swaps",
                    "Pickup / unloading",
                    "Waiting for food",
                    "Repositioning",
                ],
                [
                    "基地待命",
                    "接餐飞行",
                    "带餐飞行",
                    "任务返航",
                    "服务任务换电",
                    "取放餐",
                    "空中等餐",
                    "基地间重调度",
                ],
            ),
            [
                f.get("idle", 0),
                f.get("pre_pickup_flight", 0),
                f.get("loaded_flight", 0),
                f.get("return_flight", 0),
                sum(
                    f.get(k, 0)
                    for k in ("pre_pickup_swap", "loaded_swap", "return_swap")
                ),
                sum(f.get(k, 0) for k in ("pickup_handling", "delivery_handling")),
                f.get("sync_wait", 0),
                sum(f.get(k, 0) for k in ("empty_flight", "empty_swap")),
            ],
            T("Drones", "架"),
        ),
        (
            T("Hourly operating cost", "每小时成本"),
            T(
                [
                    "First/last-mile couriers",
                    "Courier-only orders",
                    "Flight + waiting",
                    "Fleet",
                    "Facilities + air routes",
                ],
                ["首末程骑手", "全程骑手", "飞行与等餐", "机队", "设施与航线"],
            ),
            [
                c["first_rider"] + c["terminal_rider"],
                c["direct_rider"],
                sum(c[k] for k in ("service_flight", "aerial_wait", "empty_flight")),
                c["fleet"],
                sum(
                    c[k]
                    for k in ("base_fixed", "locker_fixed", "swap_fixed", "link_fixed")
                ),
            ],
            T("CNY per hour", "元/小时"),
        ),
        (
            T("Locker configuration", "柜数量配置"),
            [str(i) for i in range(1, int(metrics["parameters"]["max_cabinets"]) + 1)],
            [
                metrics["locker_capacity"]["count_distribution"].get(i, 0)
                for i in range(1, int(metrics["parameters"]["max_cabinets"]) + 1)
            ],
            T("Sites (vertical labels: cabinets per site)", "柜点数（纵轴：每点柜数）"),
        ),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, (title, names, values, unit), color in zip(
        axs.flat, datasets, ["#187c91", "#527c97", "#99749c", "#72977f"]
    ):
        top = max(max(values), 1)
        bars = ax.barh(np.arange(len(values)), values, color=color, height=0.62)
        for bar, v in zip(bars, values):
            ax.text(
                v + 0.022 * top,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}" if top > 1000 else f"{v:.2f}",
                va="center",
                fontsize=10,
            )
        ax.set_yticks(np.arange(len(values)), names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.3)
        ax.set_xlabel(unit, fontsize=10)
        ax.set_title(title, loc="left", fontsize=14, pad=10)
        _grid(ax)
    fig.suptitle(
        T(f"H{H} | Network operations", f"H{H}｜网络运营统计"),
        fontsize=20,
        ha="left",
        x=0.025,
    )
    directory = out / f"figures_{lang}"
    directory.mkdir(exist_ok=True)
    _save(fig, directory, f"H{H}_operations")
    trace = [
        x
        for x in case["result"].get("trace", [])
        if x.get("lower", 0) > 0 and np.isfinite(x.get("upper", np.inf))
    ]
    if trace:
        elapsed = [x["elapsed"] / 60 for x in trace] + [metrics["solver"]["wall"] / 60]
        upper = [x["upper"] for x in trace] + [metrics["solver"]["objective"]]
        lower = [x["lower"] for x in trace] + [metrics["solver"]["bound"]]
        gaps = [100 * max(0, u - l) / abs(u) for u, l in zip(upper, lower)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        axes[0].step(
            elapsed,
            upper,
            where="post",
            color="#176b87",
            label=T("Incumbent", "最好可行解"),
            lw=2,
        )
        axes[0].step(
            elapsed,
            lower,
            where="post",
            color="#cf9050",
            label=T("Full-model lower bound", "完整模型下界"),
            lw=2,
        )
        axes[0].set_ylabel(T("CNY per hour", "元/小时"))
        axes[0].legend(frameon=False)
        axes[1].step(elapsed, gaps, where="post", color="#527c97", lw=2)
        axes[1].scatter(elapsed[-1], gaps[-1], c="#527c97", s=24)
        axes[1].set_ylabel(T("Full-model optimality gap (%)", "完整模型Gap（%）"))
        for ax in axes:
            ax.set_xlabel(T("Elapsed time (min)", "运行时间（分钟）"))
            _grid(ax, "y")
        fig.suptitle(
            T(
                f"H{H} | Search progress after the first positive bound",
                f"H{H}｜获得首个正下界后的搜索进展",
            ),
            fontsize=15,
        )
        _save(fig, directory, f"H{H}_convergence")


def solver_stage_figure(metrics, out, lang):
    """Plot recorded stage durations and their own allocated budgets."""
    from .solver_text import stage_name

    solver = metrics["solver"]
    rows = [
        row
        for row in solver.get("stage_rows", [])
        if row.get("wall") is not None and np.isfinite(row["wall"])
    ]
    if not rows:
        return
    translate = lambda en, cn: _tr(en, cn, lang)
    fig, ax = plt.subplots(
        figsize=(11.8, max(3.6, 0.65 * len(rows) + 2.2)), layout="constrained"
    )
    y = np.arange(len(rows))
    durations = [row["wall"] for row in rows]
    ax.barh(
        y,
        durations,
        color="#246187",
        height=0.52,
        label=translate("Actual stage time", "阶段实际时长"),
    )
    budgeted = [
        (index, row["budget"])
        for index, row in enumerate(rows)
        if row.get("budget") is not None and np.isfinite(row["budget"])
    ]
    if budgeted:
        ax.scatter(
            [budget for _, budget in budgeted],
            [index for index, _ in budgeted],
            marker="|",
            s=200,
            linewidths=2.0,
            color="#b27a27",
            label=translate("Allocated stage budget", "阶段分配预算"),
            zorder=5,
        )
    maximum = max(durations + [budget for _, budget in budgeted] + [1.0])
    for index, row in enumerate(rows):
        label = f'{row["wall"]:.2f} s'
        if row.get("budget") is not None:
            label += translate(
                f'  (budget {row["budget"]:.2f} s)', f'  （预算{row["budget"]:.2f}秒）'
            )
        ax.text(
            max(row["wall"], row.get("budget") or 0.0) + 0.025 * maximum,
            index,
            label,
            va="center",
            fontsize=10,
        )
    ax.set_yticks(y, [stage_name(row["name"], lang) for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.68 * maximum)
    ax.set_xlabel(translate("Seconds", "秒"))
    ax.legend(frameon=False, loc="lower right")
    _grid(ax)
    title = translate(
        f'{metrics["case"]} | Solver stages', f'{metrics["case"]}｜求解阶段'
    )
    subtitle = translate(
        f'Elapsed solve time: {solver["wall"]:.2f} s | Threads: {solver.get("threads")} | Seed: {solver.get("seed")}',
        f'求解经过时长：{solver["wall"]:.2f}秒｜线程：{solver.get("threads")}｜随机种子：{solver.get("seed")}',
    )
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=13, pad=14)
    directory = out / f"figures_{lang}"
    directory.mkdir(exist_ok=True)
    _save(fig, directory, f'{metrics["case"]}_solver_stages')
