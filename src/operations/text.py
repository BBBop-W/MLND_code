"""Readable summary of independently validated operations."""

from __future__ import annotations

import json

_COST_NAMES = dict(
    base_fixed="基地固定费",
    locker_fixed="柜固定费（容量档位及选址）",
    swap_fixed="换电站固定费",
    link_fixed="航线固定费",
    fleet="机队费用",
    first_rider="第一程骑手",
    terminal_rider="末端骑手",
    service_flight="服务飞行",
    aerial_wait="空中等餐",
    empty_flight="空载飞行",
    direct_rider="纯骑手配送",
)
_FLEET_NAMES = dict(
    idle="基地待命",
    pre_pickup_flight="前端飞行",
    pre_pickup_swap="前端换电",
    loaded_flight="带餐飞行",
    loaded_swap="带餐阶段换电",
    return_flight="返回飞行",
    return_swap="返回途中换电",
    pickup_handling="取餐操作",
    delivery_handling="放餐操作",
    sync_wait="空中等餐",
    empty_flight="重调度飞行",
    empty_swap="重调度途中换电",
    fleet_slack="机队约束余量",
)


def render_markdown(payload):
    s, tables = payload["summary"], payload["tables"]
    a = s["time_averages_minutes"]
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    lines = [
        "# 最新模型运营核验",
        "",
        f"实例：{s['name']}。独立核验：{'通过' if s['checks']['passed'] else '未通过'}。",
        "",
        f"总成本 **{s['objective']:.2f} 元/小时**；无人机服务 **{s['served']:.2f} 单/小时**，占潜在订单 {100*s['served_fraction']:.2f}%。",
        f"开放基地 {s['opened']['bases']} 个、柜位置 {s['opened']['lockers']} 个，共 {s['cabinet_total']:.0f} 个柜、换电站 {s['opened']['swaps']} 个；开放航线 {s['open_links']} 条，实际使用 {s['used_links']} 条。",
        f"机队 Γ={s['fleet']:.3f}；占用合计 {s['fleet_occupancy']:.3f}。跨基地服务 {s['cross_base_service']:.2f} 单/小时；空载重调度 {s['empty_flow']:.2f} 架次/小时。",
        "",
        "## 时间（从骑手取到餐开始，分钟）",
        "",
        "| 指标 | 服务订单流量加权均值 |",
        "|---|---:|",
        f"| 第一程骑手 | {fmt(a.get('first_rider'))} |",
        f"| 无人机到取餐柜（含途中换电） | {fmt(a.get('prefix'))} |",
        f"| 无人机空中等餐 | {fmt(a.get('wait'))} |",
        f"| 无人机完整任务 | {fmt(a.get('time'))} |",
        f"| 给定后半程（含呼唤骑手） | {fmt(a.get('terminal'))} |",
        f"| 其中呼唤骑手常数 | {fmt(s['terminal_dispatch_minutes'])} |",
        f"| 配送完成（含给定后半程常数） | {fmt(a.get('mean_delivery'))} |",
        f"| 相同 OD 的纯骑手取餐后配送 | {fmt(s['direct_postfetch_average_minutes'])} |",
        f"| 相同 OD 历史观测取餐后配送（有观测子集） | {fmt(s['observed_postfetch_average_minutes'])} |",
        "",
        f"所有使用服务路径的最小联合可靠性：{s['min_joint_reliability']:.6f}。",
        "",
        "## 基地运营",
        "",
        "| 基地 | 开放 | ρ | 待命无人机 | 服务出发/h | 服务返回/h | 空载出发/h | 空载返回/h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["bases"]:
        lines.append(
            "| "
            + " | ".join(
                [str(row["base"]), str(row["open"])]
                + [
                    fmt(row[k])
                    for k in (
                        "rho",
                        "idle",
                        "service_out",
                        "service_in",
                        "empty_out",
                        "empty_in",
                    )
                ]
            )
            + " |"
        )
    lines += ["", "## 无人机占用", "", "| 状态 | 平均无人机数量 |", "|---|---:|"]
    lines += [
        f"| {_FLEET_NAMES.get(r['component'], r['component'])} | {r['drones']:.4f} |"
        for r in tables["fleet"]
    ]
    lines += ["", "## 成本", "", "| 项目 | 元/小时 |", "|---|---:|"]
    lines += [
        f"| {_COST_NAMES.get(r['component'], r['component'])} | {r['cny_per_hour']:.4f} |"
        for r in tables["costs"]
    ]
    if any(r["cny_per_hour"] > 0 for r in tables.get("site_costs", [])):
        lines += [
            "",
            "上述设施固定费中包含以下按开放位置计取的选址费用：",
            "",
            "| 设施 | 开放位置数 | 选址费（元/小时） |",
            "|---|---:|---:|",
        ]
        names = dict(base="基地", locker="柜位置", swap="换电站")
        lines += [
            f"| {names[r['component']]} | {r['opened_locations']} | {r['cny_per_hour']:.4f} |"
            for r in tables["site_costs"]
        ]
    lines += [
        "",
        "## 核验",
        "",
        "按实际节点序列重建飞行、换电、操作、随机等待、电池区间及配送分位数，并核对订单分配、基地守恒、柜容量、航线、机队与完整成本。",
        "",
        "末端时间采用给定均值；无人机活动在返回基地时结束，未额外追加基地补能时间。",
        "",
    ]
    if s["terminal_calibration"]:
        lines += [
            "末端时间校准：",
            "",
            "```json",
            json.dumps(s["terminal_calibration"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "```json",
        json.dumps(s["checks"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)
