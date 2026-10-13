"""Collect frozen experiment results without changing the original run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif not isinstance(item, list):
            result[name] = item
    return result


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


TOKENS = {
    "case": "算例", "case_id": "算例编号", "round": "轮次", "factor": "变化因素", "multiplier": "倍率",
    "method": "方法", "OD": "OD对数", "candidate": "候选", "candidates": "候选",
    "bases": "基地", "base": "基地", "lockers": "柜站", "locker": "柜站", "swaps": "换电站", "swap": "换电",
    "fleet": "机队", "drones": "无人机数", "idle": "待命", "opened": "开放", "open": "开放",
    "components": "组成", "total": "总计", "demand": "需求", "served": "无人机服务", "share": "占比",
    "pickup": "到柜自取", "home": "到家", "direct": "纯骑手", "repositioning": "空载重调度",
    "flow": "流量", "capacity": "处理能力", "locker_capacity": "柜处理能力", "utilization": "利用率", "time": "时间", "times": "时长",
    "cost": "成本", "costs": "成本", "parameters": "参数", "scenario_settings": "情景参数",
    "solver": "求解", "params": "求解参数", "raw": "求解器原始记录", "operations": "运营补充",
    "validation": "校验", "independent_checks": "独立校验", "reconciliation": "对账",
    "per_hour": "每小时", "per_km_hour": "每公里每小时", "minutes": "分钟", "hours": "小时",
    "pct": "百分比", "fraction": "比例", "mean": "均值", "median": "中位数", "p90": "90分位",
    "min": "最小值", "max": "最大值", "count": "数量", "epsilon": "失效概率",
    "speed": "速度", "battery_time": "续航小时", "battery": "续航距离", "swap_time": "换电时长",
    "unit_capacity": "单柜处理能力", "max_cabinets": "单点柜单元上限", "fleet_cost": "无人机小时固定成本",
    "base_cost": "基地小时固定成本", "hub_cost": "单柜小时固定成本", "swap_cost": "换电站小时固定成本",
    "base_site_cost": "基地选址小时成本", "hub_site_cost": "柜站选址小时成本", "swap_site_cost": "换电站选址小时成本",
    "airline_hh_cost_per_km_hour": "柜柜航线小时每公里固定成本",
    "airline_connection_cost_per_km_hour": "连接航线小时每公里固定成本", "drone_time_cost": "飞行小时运行成本",
    "hover_cost": "等待小时运行成本", "pre_pickup_flight": "取餐前飞行", "pre_pickup_swap": "取餐前换电",
    "loaded_flight": "带载飞行", "loaded_swap": "带载换电", "return_flight": "回程飞行", "return_swap": "回程换电",
    "pickup_handling": "取餐装载", "delivery_handling": "投递卸载", "sync_wait": "会合等待", "fleet_slack": "机队余量",
    "mission_occupancy": "配送任务占用", "empty_occupancy": "空载重调度占用", "first_rider": "首段骑手",
    "prefix": "无人机到取餐柜过程", "wait": "会合等待", "terminal": "末端交付", "mean_delivery": "配送均时",
    "quantile_delivery": "配送分位时长", "fly": "全部飞行", "direct_same_served_OD": "同一已服务OD纯骑手均时",
    "OD_mean_delivery": "逐OD配送均时分布", "OD_mean_courier": "逐OD纯骑手均时分布",
    "fraction_at_least_5min_faster": "比纯骑手快至少5分钟的服务流占比",
    "first_leg": "首段骑手", "distance_m": "距离米", "colocated_within_1m_flow_fraction": "首段距离不超1米的服务流占比",
    "swapping": "换电", "service_visits_per_hour": "配送换电次数每小时", "reposition_visits_per_hour": "重调度换电次数每小时",
    "service_flow_with_swap_fraction": "使用换电配送流占比", "base_fixed": "基地固定成本", "locker_fixed": "柜固定成本",
    "swap_fixed": "换电固定成本", "link_fixed": "航线固定成本", "terminal_rider": "末段骑手成本",
    "service_flight": "配送飞行成本", "aerial_wait": "等待成本", "empty_flight": "空载飞行成本", "direct_rider": "纯骑手成本",
    "full_sites": "容量满载柜站数", "at_least_95pct_sites": "利用率至少95%的柜站数", "handling_per_hour": "柜处理流量每小时",
    "count_distribution": "开放柜单元数量分布", "assigned_potential_per_hour": "已分配潜在订单每小时",
    "unassigned_potential_per_hour": "未分配潜在订单每小时", "unavailable_assigned_per_hour": "已分配但无人机不可用订单每小时",
    "cross_base_service_per_hour": "跨基地配送每小时", "mean_duration_minutes": "平均持续分钟",
    "wall": "墙钟秒", "recorded_wall": "记录墙钟秒", "total_wall": "含后处理墙钟秒", "direct_wall": "直接求解墙钟秒",
    "budget": "时限秒", "direct_budget": "直接求解时限秒", "bound": "最优下界", "objective": "目标值",
    "gap": "相对求解差距", "status": "求解状态码", "termination_reason": "终止原因", "threads": "求解线程数",
    "seed": "随机种子", "mip_gap": "相对差距目标", "mip_gap_abs": "绝对差距目标",
    "state_nodes": "状态图节点数", "state_arcs": "状态图弧数", "service_graphs": "配送状态图数", "empty_graphs": "空载状态图数",
    "construction_complete": "模型构建完成", "min_joint_reliability": "使用路径最小联合可靠性",
    "rho_open_min": "开放基地rho最小值", "rho_open_mean": "开放基地rho均值", "rho_open_max": "开放基地rho最大值",
    "reliability_mean_flow_weighted": "服务流加权路径联合可靠性", "served_OD_count": "有无人机服务的OD数",
    "all_OD_direct_mean_minutes": "全体OD需求加权纯骑手均时", "system_delivery_mean_minutes": "全部订单加权模型配送均时",
}


LABELS = {
    "case": "算例编号", "total_cost_per_hour": "系统总成本", "all_courier_cost_per_hour": "全部骑手基准成本",
    "saving_per_hour": "每小时节省成本", "saving_pct": "成本节省率", "drone_served_per_hour": "无人机服务订单量",
    "drone_served_share_pct": "无人机服务占比", "pickup_per_hour": "到柜自取订单量", "home_per_hour": "无人机接驳到家订单量",
    "direct_courier_per_hour": "纯骑手订单量", "fleet": "连续机队规模", "locker_units": "安装柜单元总数",
    "opened_airlines": "开放有向连接总数", "used_airlines": "实际使用有向连接总数", "cross_base_service_pct": "跨基地配送占比",
    "swap_visits_per_hour": "总换电访问次数", "locker_capacity_utilization_pct": "总体柜处理能力利用率",
    "mean_delivery_minutes": "无人机服务订单平均配送时长", "direct_same_served_OD_minutes": "同一无人机服务OD纯骑手均时",
    "gap_pct": "最终求解Gap", "solver_seconds": "求解墙钟时间", "validation_passed": "结果校验通过",
    "solver_objective": "求解器目标值", "solver_gap": "求解器相对Gap", "wall_minutes": "求解墙钟分钟",
    "above_gap_target": "最终Gap超过0.5%", "saving_upper_bound_pct": "由成本下界得到的节省率上界",
    "cost_per_order": "每潜在订单平均成本", "fleet_per_100_drone_orders": "每小时100单无人机订单所需机队",
    "area_km2": "研究区域面积", "width_km": "区域宽度", "height_km": "区域高度",
    "demand_per_hour": "总潜在需求", "run_path": "原始算例结果路径", "job_id": "实验任务编号",
    "candidate_bases": "候选基地数量", "candidate_lockers": "候选柜站数量", "candidate_swaps": "候选换电站数量",
    "opened_bases": "开放基地数量", "opened_lockers": "开放柜站数量", "opened_swaps": "开放换电站数量",
    "reposition_flow_per_hour": "空载重调度流量", "swap_service_visits_per_hour": "配送换电访问次数",
    "swap_reposition_visits_per_hour": "重调度换电访问次数", "service_using_swap_pct": "使用换电的配送流占比",
    "scenario": "情景标识", "solver_termination_reason": "求解终止原因",
}

PARAMETER_LABELS = {
    "name": "名称", "source": "来源", "cost_period": "成本计费周期", "choice": "选择模型", "input": "实际输入",
    "config": "配置", "n_l": "候选基地数", "n_h": "候选柜站数", "n_k": "候选换电站数", "n_od": "OD对数",
    "terminal_dispatch_time": "末端呼唤骑手平均小时", "pickup_handling_time": "装载小时", "delivery_handling_time": "卸载小时",
    "rider_time_family": "骑手时长分布", "rider_slope_minutes_per_straight_km": "骑手每直线公里分钟数",
    "rider_speed": "骑手速度", "road_factor": "路网距离系数", "rider_time_cv": "骑手时间变异系数",
    "dispatch_clock": "派遣触发时刻", "pre_dispatch_mean_hours": "订单至取餐平均小时（未计入配送时钟）",
    "airline_cost_multiplier": "航线成本倍率", "flight_cost": "每公里飞行成本", "c0": "全部骑手基准成本",
    "demand_signature": "需求数据校验签名", "model_version": "模型版本", "locker_asc": "自取效用常数",
    "delivery_time_coefficient": "配送时间效用系数", "price_coefficient": "价格效用系数",
    "walk_distance_coefficient": "步行距离敏感系数", "home_eta_minutes": "到家展示分钟", "locker_eta_minutes": "到柜展示分钟",
    "locker_discount_cny": "自取优惠金额", "max_walk_km": "最大步行距离", "amortization_years": "折旧年数",
    "operating_days_per_year": "每年运营天数", "operating_hours_per_day": "每天运营小时",
    "rider_initial_charge_cny": "骑手起步费", "rider_included_distance_km": "骑手起步费包含距离",
    "rider_incremental_cost_per_km": "骑手超出距离每公里费用", "road_distance_factor": "路网距离系数",
    "feeder_rider_cost_factor": "接驳骑手费用折算系数", "first_leg_cost": "首段骑手成本定义", "last_leg_cost": "末段骑手成本定义",
    "base_site_investment": "基地选址一次性投资", "locker_site_investment": "柜站选址一次性投资", "swap_site_investment": "换电选址一次性投资",
    "flight_per_hour": "每架飞行小时费用", "hover_per_hour": "每架等待小时费用", "rider_initial_charge": "骑手起步费",
    "rider_additional_per_km": "骑手超出距离每公里费用", "feeder_rider_multiplier": "接驳骑手费用折算系数",
    "base_investment": "基地建设原值", "cabinet_investment": "单柜建设原值", "swap_investment": "换电站建设原值",
    "drone_investment": "单架无人机原值", "directed_route_investment_per_km": "单向航线每公里建设原值",
    "cabinet_capacity": "单柜处理能力", "drone_speed_kmh": "无人机飞行速度", "battery_endurance_minutes": "电池续航分钟",
    "swap_minutes": "单次换电分钟", "pickup_minutes": "装载分钟", "unloading_minutes": "卸载分钟",
    "failure_probability": "失效概率", "rider_minutes_per_km": "骑手每公里分钟数", "terminal_dispatch_minutes": "末端呼唤骑手平均分钟",
    "deadline_multiplier": "配送时限倍率", "depreciation": "折旧", "years": "年数", "days_per_year": "每年天数",
    "hours_per_day": "每天小时数", "demand_scale": "需求倍率", "empty_flight": "空载飞行", "empty_swap": "空载换电",
}
TOKENS.update(PARAMETER_LABELS)

PARAMETER_UNITS = {
    "unit_capacity": "次/(小时·柜单元)", "cabinet_capacity": "次/(小时·柜单元)", "max_cabinets": "柜单元/位置",
    "base_cost": "元/(基地·小时)", "hub_cost": "元/(柜单元·小时)", "swap_cost": "元/(换电站·小时)",
    "base_site_cost": "元/(位置·小时)", "hub_site_cost": "元/(位置·小时)", "swap_site_cost": "元/(位置·小时)",
    "fleet_cost": "元/(架·小时)", "drone_time_cost": "元/(架·小时)", "hover_cost": "元/(架·小时)",
    "flight_per_hour": "元/(架·小时)", "hover_per_hour": "元/(架·小时)",
    "airline_hh_cost_per_km_hour": "元/(km·小时)", "airline_connection_cost_per_km_hour": "元/(km·小时)",
    "speed": "km/h", "rider_speed": "km/h", "drone_speed_kmh": "km/h", "flight_cost": "元/km",
    "battery_time": "小时", "swap_time": "小时", "terminal_dispatch_time": "小时", "pickup_handling_time": "小时",
    "delivery_handling_time": "小时", "pre_dispatch_mean_hours": "小时", "battery": "km",
    "rider_slope_minutes_per_straight_km": "分钟/km（直线）", "rider_minutes_per_km": "分钟/km",
    "rider_initial_charge_cny": "元/单", "rider_initial_charge": "元/单", "rider_incremental_cost_per_km": "元/km",
    "rider_additional_per_km": "元/km", "rider_included_distance_km": "km", "locker_discount_cny": "元/单",
    "walk_distance_coefficient": "1/km", "delivery_time_coefficient": "1/分钟", "price_coefficient": "1/元",
    "max_walk_km": "km", "home_eta_minutes": "分钟", "locker_eta_minutes": "分钟", "battery_endurance_minutes": "分钟",
    "swap_minutes": "分钟", "pickup_minutes": "分钟", "unloading_minutes": "分钟", "terminal_dispatch_minutes": "分钟",
    "amortization_years": "年", "operating_days_per_year": "天/年", "operating_hours_per_day": "小时/天",
    "years": "年", "days_per_year": "天/年", "hours_per_day": "小时/天", "c0": "元/小时",
    "base_site_investment": "元/位置", "locker_site_investment": "元/位置", "swap_site_investment": "元/位置",
    "base_investment": "元/基地", "cabinet_investment": "元/柜单元", "swap_investment": "元/换电站", "drone_investment": "元/架",
    "directed_route_investment_per_km": "元/km", "first_leg_cost": "文本定义", "last_leg_cost": "文本定义",
    "epsilon": "0–1", "failure_probability": "0–1", "rider_time_cv": "无量纲", "road_factor": "无量纲",
    "airline_cost_multiplier": "倍率", "demand_scale": "倍率", "deadline_multiplier": "倍率", "feeder_rider_multiplier": "倍率",
    "feeder_rider_cost_factor": "倍率", "road_distance_factor": "无量纲", "locker_asc": "无量纲效用",
    "n_l": "个", "n_h": "个", "n_k": "个", "n_od": "对",
}


def describe(field):
    parts = field.split("__")
    leaf = parts[-1]
    if field in LABELS:
        label = LABELS[field]
    else:
        label = "／".join(TOKENS.get(part, part.replace("_", " ")) for part in parts)
    if field.startswith("fleet_") and not field.startswith("fleet_per"):
        label = "机队占用／" + TOKENS.get(field.removeprefix("fleet_"), field.removeprefix("fleet_"))
    if field.startswith("cost_") and field.endswith("_per_hour"):
        label = "成本组成／" + TOKENS.get(field.removeprefix("cost_").removesuffix("_per_hour"), field)
    unit = "—"
    if "pct" in leaf:
        unit = "%"
    elif "fraction" in leaf or leaf in {"gap", "mip_gap", "served_share", "utilization", "rho_open_min", "rho_open_mean", "rho_open_max", "min_joint_reliability", "reliability_mean_flow_weighted"}:
        unit = "0–1"
    elif "cost" in field or "cny" in field or (parts[0] in {"costs", "raw", "solver"} and leaf in {"total", "objective", "bound", "recourse_objective", "all_ground_cost"}):
        unit = "元/小时"
    elif "minutes" in field or parts[0] == "times":
        unit = "分钟"
    elif leaf in {"wall", "recorded_wall", "total_wall", "budget", "direct_wall", "direct_budget", "solver_seconds"}:
        unit = "秒"
    elif "per_hour" in leaf:
        unit = "单/小时（访问字段为次/小时）"
    elif parts[0] == "fleet" or field.startswith("fleet_") or field == "fleet":
        unit = "架"
    elif leaf in {"speed", "rider_speed", "drone_speed_kmh"}:
        unit = "km/h"
    elif "km2" in field:
        unit = "km²"
    elif leaf.endswith("_km") or leaf == "battery":
        unit = "km"
    elif leaf.endswith("_time") or leaf.endswith("_hours"):
        unit = "小时"
    elif "investment" in leaf:
        unit = "元（五年期原值）"
    elif any(x in leaf for x in ["count", "graphs", "arcs", "nodes", "opened", "candidate"]) or leaf in {"OD", "max_cabinets", "cabinet_total", "full_sites", "at_least_95pct_sites"}:
        unit = "个"
    if parts[0] in {"parameters", "scenario_settings", "input", "config"}:
        unit = PARAMETER_UNITS.get(leaf, "—")
    if field in {"cost_per_order", "costs__per_potential_order"}:
        unit = "元/单"
    if field in {"fleet_per_100_drone_orders"}:
        unit = "架/(100单/小时)"
    if field.startswith("first_leg__distance_m"):
        unit = "米"
    if field.startswith("locker_capacity__count_distribution"):
        unit = "位置数"
    if field in {"operations__fleet_occupancy", "raw__fleet"}:
        unit = "架"
    if field in {"demand__pickup", "demand__home", "demand__direct", "raw__served", "raw__demand"}:
        unit = "单/小时"
    if field in {"saving_per_hour", "bound", "solver_objective"}:
        unit = "元/小时"
    if field in {"solver_gap"}:
        unit = "0–1"
    if field in {"max_cabinets", "walk_distance_coefficient"}:
        unit = PARAMETER_UNITS[field]
    source = "sensitivity/sensitivity_metrics.json"
    if parts[0] == "input":
        source = "input_metadata.json: values"
    elif parts[0] == "config":
        source = "config.json: model_config（仅成本、运营、选择、折旧、需求倍率）"
    elif parts[0] == "raw":
        source = "result.json"
    elif parts[0] == "operations":
        source = "operations.json: summary"
    elif leaf.startswith("rho_open") or leaf == "reliability_mean_flow_weighted" or leaf in {"served_OD_count", "all_OD_direct_mean_minutes", "system_delivery_mean_minutes"}:
        source = "operations.json: tables，加权或汇总计算"
    elif field in {"case_id", "round", "case", "factor", "multiplier", "job_id", "run_path", "scenario"}:
        source = "对应轮次batch计划、保存情景标签及all_results.json索引"
    note = "缺失保留为空；原始数值精度。"
    if field.startswith("times__") or field in {"mean_delivery_minutes", "direct_same_served_OD_minutes", "all_OD_direct_mean_minutes", "system_delivery_mean_minutes", "operations__direct_postfetch_average_minutes", "operations__observed_postfetch_average_minutes"}:
        note += "配送时钟从骑手取到餐开始；不含商家备餐和接单前时间；末端使用统一到家口径。"
    if field.startswith(("times__OD_mean_delivery__", "times__OD_mean_courier__")):
        note += "此处为按服务流加权的OD均时分布，非单笔随机配送时间分布。"
    if field == "times__quantile_delivery":
        note += "按服务流加权的路径分位配送时长均值。"
    if parts[0] == "solver" and any(x in field for x in ["benders", "polish", "full_cg", "refinement", "heuristic"]):
        note += "历史通用报告字段；本批为arc法，零或空值不表示执行过该阶段。"
    if field.endswith("open_airlines") and field.startswith("raw"):
        note += "只含原始航线分类；连接线另记open_connectors，二者合计为总开放连接数。"
    return {"field": field, "中文名称": label, "单位": unit, "来源": source, "说明": note}


def build(output):
    output = Path(output)
    source_rows = read(output / "all_results.json")
    rows, all_details, tables = [], [], defaultdict(list)
    for source_row in source_rows:
        run = Path(source_row["run_path"])
        report = read(run / "sensitivity/sensitivity_metrics.json")
        detail = report["details"]
        operations = read(run / "operations.json")
        raw = read(run / "result.json")
        metadata = read(run / "input_metadata.json")["values"]
        config = read(run / "config.json")
        identity = {"case_id": source_row["case"], "round": source_row["round"], "factor": source_row["factor"], "multiplier": source_row["multiplier"]}
        row = {**identity, **source_row}
        for group in ("parameters", "scenario_settings", "demand", "fleet", "repositioning", "locker_capacity", "times", "first_leg", "swapping", "costs", "solver", "validation"):
            row.update(flatten(detail.get(group, {}), group))
        row.update({f"input__{k}": v for k, v in metadata.items() if not isinstance(v, (dict, list))})
        model_config = config.get("model_config", {})
        for group in ("costs", "operations", "choice", "depreciation"):
            row.update(flatten(model_config.get(group, {}), f"config__{group}"))
        row["config__demand_scale"] = model_config.get("data", {}).get("demand_scale")
        row.update({f"raw__{k}": v for k, v in raw.items() if not isinstance(v, (dict, list))})
        row.update(flatten(raw.get("params", {}), "raw__params"))
        row.update(flatten(raw.get("budget", {}) if isinstance(raw.get("budget"), dict) else {}, "raw__budget"))
        osum = operations["summary"]
        for key in ("min_joint_reliability", "fleet_occupancy", "direct_postfetch_average_minutes", "observed_postfetch_average_minutes", "terminal_dispatch_minutes"):
            row[f"operations__{key}"] = osum.get(key)
        for key in ("area_km2", "width_km", "height_km"):
            row[key] = report["maps"].get(key)
        bases = operations["tables"]["bases"]
        rho = [v["rho"] for v in bases if v["open"]]
        row.update(rho_open_min=min(rho) if rho else None, rho_open_mean=sum(rho) / len(rho) if rho else None, rho_open_max=max(rho) if rho else None)
        ods = operations["tables"]["od_times"]
        row["served_OD_count"] = sum(v["served_per_hour"] > 1e-7 for v in ods)
        row["all_OD_direct_mean_minutes"] = sum(v["demand_per_hour"] * v["direct_postfetch_minutes"] for v in ods) / row["demand_per_hour"]
        row["system_delivery_mean_minutes"] = sum(v["direct_per_hour"] * v["direct_postfetch_minutes"] + v["served_per_hour"] * (v["mean_delivery_minutes"] or 0) for v in ods) / row["demand_per_hour"]
        paths = operations["tables"]["paths"]
        spaths = [v for v in paths if v["kind"] == "service"]
        row["reliability_mean_flow_weighted"] = sum(v["flow_per_hour"] * v["joint_reliability"] for v in spaths) / sum(v["flow_per_hour"] for v in spaths) if spaths else None
        for name, values in operations["tables"].items():
            tables[name].extend({**identity, **value} for value in values)
        tables["segments"].extend({**identity, **value} for value in operations["segments"])
        tables["swap_stations"].extend({**identity, **value} for value in detail["swapping"]["stations"])
        for origin_base, service_row in enumerate(detail["repositioning"]["service_base_pair_matrix"]):
            for destination_base, service_flow in enumerate(service_row):
                tables["base_pairs"].append({**identity, "origin_base": origin_base, "destination_base": destination_base, "service_per_hour": service_flow, "reposition_per_hour": detail["repositioning"]["base_pair_matrix"][origin_base][destination_base]})
        tables["site_investment"].extend({**identity, **v} for v in detail.get("site_costs", {}).get("rows", []))
        all_details.append({**identity, "run_path": str(run), "details": detail, "operations": operations, "raw_result": raw, "config": config, "input_scalar_parameters": {k: v for k, v in metadata.items() if not isinstance(v, (dict, list))}, "source_hashes": report["sources"]})
        assert abs(sum(v["cny_per_hour"] for v in operations["tables"]["costs"]) - row["total_cost_per_hour"]) < 1e-5
        assert len(ods) == row["OD"]
        assert all(v is None or not isinstance(v, float) or math.isfinite(v) for v in row.values())
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    rows = [{key: row.get(key) for key in columns} for row in rows]
    output.joinpath("full_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    write_csv(output / "full_metrics.csv", rows, columns)
    dictionary = [describe(field) for field in columns]
    write_csv(output / "metric_dictionary.csv", dictionary)
    output.joinpath("metric_dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=False, indent=2))
    output.joinpath("all_details.json").write_text(json.dumps(all_details, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    output.joinpath("details").mkdir(exist_ok=True)
    for name, records in tables.items():
        write_csv(output / "details" / f"{name}.csv", records)
    # Compact JSON authoring input keeps all native scalar types for Excel.
    included = ("bases", "lockers", "swap_stations", "base_pairs", "costs", "fleet", "site_investment", "od_times", "paths")
    output.joinpath("workbook_data.json").write_text(json.dumps({"rows": rows, "dictionary": dictionary, "tables": {k: tables[k] for k in included}}, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    info = {"cases": len(rows), "metrics": len(columns), "details": {k: len(v) for k, v in tables.items()}, "rounds": sorted({v["round"] for v in rows}), "all_cost_reconciliations_passed": True, "all_OD_counts_match": True}
    output.joinpath("data_validation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
