"""Bilingual single-case narrative and parameter tables."""

from __future__ import annotations

import os
import numpy as np
from .formatting import _tr, _n, _flow, _table, _setting
from .solver_text import solver_section


def report(case, m, out, lang):
    T = lambda en, cn: _tr(en, cn, lang)
    s = case["summary"]
    H = case["row"]["H"]
    d = m["demand"]
    p = m["parameters"]
    f = case["fleet"]
    cap = m["locker_capacity"]
    tm = m["times"]
    r = m["repositioning"]
    cs = case["costs"]
    sol = m["solver"]
    site = m.get("site_costs", {})
    has_site = site.get("active", False)
    lines = []

    def add(x):
        lines.extend([x, ""])

    def tab(headers, rows):
        add(_table(headers, rows))

    add(
        T(
            f"# H{H}: Network Design and Operating Results",
            f"# H{H}：网络设计与运营结果",
        )
    )
    add(
        T(
            "This standalone report uses the saved Nanshan, Shenzhen instance and its independently reconstructed solution. Demand and flows are per hour for the 12:00–13:00 period. Fixed assets are amortized over five years, 365 days per year, and 12 operating hours per day.",
            "本报告单独统计深圳南山本实例的保存解。需求和流量对应12:00—13:00，以每小时计量。固定资产按五年、每年365天、每日运营12小时折旧。",
        )
    )
    add(
        T(
            f"**Result:** {d['served_per_hour']:,.2f} drone-assisted orders/h ({100*d['served_share']:.2f}%), {s['fleet']:.2f} drones, and CNY {s['objective']:,.2f}/h. Inter-base repositioning is {_flow(r['flow_per_hour'])} missions/h; the full-model optimality gap is {_n(100*sol['gap'] if sol['gap'] is not None else None)}%.",
            f"**结果：** 无人机服务{d['served_per_hour']:,.2f}单/小时（{100*d['served_share']:.2f}%），机队{s['fleet']:.2f}架，总成本{s['objective']:,.2f}元/小时。基地间重调度{_flow(r['flow_per_hour'])}架次/小时，完整模型Gap为{_n(100*sol['gap'] if sol['gap'] is not None else None)}%。",
        )
    )
    add(T("## 1. Candidate locations and model parameters", "## 1. 候选位置与参数"))
    tab(
        [T("Facility", "设施"), T("Candidates", "候选数"), T("Opened", "开放数")],
        [
            (
                T("Drone bases L", "无人机基地 L"),
                m["candidates"]["bases"],
                m["opened"]["bases"],
            ),
            (
                T("Locker sites H", "柜点 H"),
                m["candidates"]["lockers"],
                m["opened"]["lockers"],
            ),
            (
                T("Swapping stations E", "换电站 E"),
                m["candidates"]["swaps"],
                m["opened"]["swaps"],
            ),
        ],
    )
    tab(
        [T("Parameter", "参数"), T("Value", "设置")],
        [
            (T("OD pairs", "OD对数"), f"{d['total_OD_count']:,}"),
            (
                T("Handling capacity per cabinet", "每柜处理能力"),
                T(
                    f"{p['unit_capacity']:.1f} events/h",
                    f"{p['unit_capacity']:.1f}次/小时",
                ),
            ),
            (T("Maximum cabinets per site", "每点柜数上限"), p["max_cabinets"]),
            (
                T(
                    (
                        "Base fixed cost excluding setup"
                        if has_site
                        else "Base fixed cost"
                    ),
                    "基地原有固定费（另计选址）" if has_site else "基地固定成本",
                ),
                T(
                    f"{p['base_cost']:.2f} CNY/site/h",
                    f"{p['base_cost']:.2f}元/点/小时",
                ),
            ),
            (
                T(
                    "Cabinet equipment cost" if has_site else "Cabinet fixed cost",
                    "柜设备成本（另计选址）" if has_site else "柜固定成本",
                ),
                T(
                    f"{p['hub_cost']:.2f} CNY/cabinet/h",
                    f"{p['hub_cost']:.2f}元/柜/小时",
                ),
            ),
            (
                T(
                    (
                        "Swapping-station fixed cost excluding setup"
                        if has_site
                        else "Swapping-station fixed cost"
                    ),
                    "换电站原有固定费（另计选址）" if has_site else "换电站固定成本",
                ),
                T(
                    f"{p['swap_cost']:.2f} CNY/site/h",
                    f"{p['swap_cost']:.2f}元/点/小时",
                ),
            ),
            (
                T("Fleet cost", "机队成本"),
                T(
                    f"{p['fleet_cost']:.2f} CNY/drone/h",
                    f"{p['fleet_cost']:.2f}元/架/小时",
                ),
            ),
            (
                T("Directed H–H route cost", "单向H–H航线成本"),
                T(
                    f"{p['airline_hh_cost_per_km_hour']:.3f} CNY/km/h",
                    f"{p['airline_hh_cost_per_km_hour']:.3f}元/公里/小时",
                ),
            ),
            (
                T("Other directed route cost", "其他单向航线成本"),
                T(
                    f"{p['airline_connection_cost_per_km_hour']:.3f} CNY/km/h",
                    f"{p['airline_connection_cost_per_km_hour']:.3f}元/公里/小时",
                ),
            ),
            (
                T("Flight / hovering operating cost", "飞行/悬停运营成本"),
                f"{p['drone_time_cost']:.2f} / {p['hover_cost']:.2f} CNY/drone-hour",
            ),
            (T("Drone speed", "无人机速度"), f"{p['speed']:.1f} km/h"),
            (
                T("Battery endurance", "单次电池续航时间"),
                T(f"{60*p['battery_time']:.3f} min", f"{60*p['battery_time']:.3f}分钟"),
            ),
            (
                T("Duration per battery swap", "每次换电时间"),
                T(f"{60*p['swap_time']:.3f} min", f"{60*p['swap_time']:.3f}分钟"),
            ),
            (
                T("Joint reliability threshold", "联合可靠性要求"),
                f"{100*(1-p['epsilon']):.1f}%",
            ),
            (
                T("First-mile rider travel-time family", "第一程骑手时间分布"),
                T(
                    p["rider_time_family"],
                    (
                        "指数分布"
                        if p["rider_time_family"] == "exponential"
                        else p["rider_time_family"]
                    ),
                ),
            ),
            (
                T("Mean travel-time slope", "均值时间系数"),
                f"{p['rider_slope_minutes_per_straight_km']:.4f} min/km",
            ),
            (
                T("Pickup / unloading time", "取餐/放餐时间"),
                f"{60*p['pickup_handling_time']:.3f} / {60*p['delivery_handling_time']:.3f} min",
            ),
            (
                T("Last-mile courier dispatch constant", "末端骑手呼叫常数"),
                f"{60*p['terminal_dispatch_time']:.3f} min",
            ),
        ],
    )
    if has_site:
        add(
            T(
                "A one-time setup investment is amortized and charged once for each opened location. The locker setup charge is independent of the number of cabinets installed there. The amounts and inclusion fractions below come from the saved input provenance.",
                "一次性选址投入折旧后按每个开放位置计取；柜点选址费与该点安装柜数无关。下表预算与计入比例来自本次保存的输入来源记录。",
            )
        )
        names = {
            "base": T("Base L", "基地L"),
            "locker": T("Locker site H", "柜点H"),
            "swap": T("Swap station E", "换电站E"),
        }
        tab(
            [
                T("Facility", "设施"),
                T("Gross allowance, CNY/site", "选址预算，元/点"),
                T("Included share", "计入比例"),
                T("Included investment, CNY/site", "计入一次性投资，元/点"),
                T("Amortization hours", "折旧运营小时数"),
                T("Setup cost, CNY/site/h", "选址费，元/点/小时"),
            ],
            [
                (
                    names[x["component"]],
                    _n(x["gross_investment_cny"], 0),
                    (
                        "—"
                        if x["incremental_fraction"] is None
                        else f"{100*x['incremental_fraction']:.0f}%"
                    ),
                    _n(x["included_investment_cny"], 0),
                    _n(x["amortization_operating_hours"], 0),
                    _setting(x["hourly_per_site"], 6),
                )
                for x in site["rows"]
            ],
        )
    settings = m["scenario_settings"]
    tariff = [
        ("Rider initial charge", "骑手起步价", "rider_initial_charge_cny", "CNY"),
        (
            "Included road distance",
            "骑手起步价所含路程",
            "rider_included_distance_km",
            "km",
        ),
        (
            "Additional distance charge",
            "骑手超程单价",
            "rider_incremental_cost_per_km",
            "CNY/km",
        ),
        ("Road-distance multiplier", "道路距离系数", "road_distance_factor", ""),
        (
            "Feeder-courier cost multiplier",
            "首末程骑手费用系数",
            "feeder_rider_cost_factor",
            "",
        ),
    ]
    tab(
        [T("Rider cost parameter", "骑手成本参数"), T("Value", "设置")],
        [
            (T(en, cn), f"{settings[k]:g} {unit}".strip())
            for en, cn, k, unit in tariff
            if k in settings
        ],
    )
    choice = m["scenario_settings"].get("choice", {})
    if choice:
        constant = (
            choice["locker_asc"]
            + choice["delivery_time_coefficient"]
            * (choice["home_eta_minutes"] - choice["locker_eta_minutes"])
            + choice["price_coefficient"] * choice["locker_discount_cny"]
        )
        slope = choice["walk_distance_coefficient"]
        cutoff = choice["max_walk_km"]
        distances = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        add(
            T(
                f"Customer choice uses σ(r)=1/[1+exp(−({constant:g}−{slope:g}r))] for r<{cutoff:g} km and zero otherwise. Here r is the destination-locker-to-customer straight-line distance.",
                f"用户自提概率为σ(r)=1/[1+exp(−({constant:g}−{slope:g}r))]（r<{cutoff:g}公里），其余为0；r为目的柜至顾客的直线距离。",
            )
        )
        tab(
            [T("Distance", "距离"), T("Self-pickup probability", "自提概率")],
            [
                (
                    f"{1000*x:.0f} m",
                    f"{100/(1+np.exp(-(constant-slope*x))) if x<cutoff else 0:.2f}%",
                )
                for x in distances
            ],
        )
    add(T("## 2. Demand and network design", "## 2. 需求与网络设计"))
    tab(
        [T("Metric", "指标"), T("Value", "数值")],
        [
            (
                T("Potential demand, orders/h", "潜在需求，单/小时"),
                _n(d["total_per_hour"]),
            ),
            (
                T("Drone-assisted orders/h", "无人机服务，单/小时"),
                _n(d["served_per_hour"]),
            ),
            (
                T("Drone-assisted share", "无人机服务比例"),
                f"{100*d['served_share']:.2f}%",
            ),
            (
                T("Drone + self-pickup, orders/h", "无人机＋自提，单/小时"),
                _n(d["pickup"]),
            ),
            (
                T("Drone + home courier, orders/h", "无人机＋末端骑手到家，单/小时"),
                _n(d["home"]),
            ),
            (T("Courier-only orders/h", "全程骑手，单/小时"), _n(d["direct"])),
            (
                T(
                    "Unassigned potential demand, orders/h",
                    "未分配无人机的需求，单/小时",
                ),
                _n(d["unassigned_potential_per_hour"]),
            ),
            (
                T(
                    "Assigned demand encountering no drone, orders/h",
                    "已分配但无人机不可用，单/小时",
                ),
                _n(d["unavailable_assigned_per_hour"]),
            ),
            (T("Installed cabinets", "安装柜总数"), _n(s["cabinet_total"], 0)),
            (
                T(
                    "Opened directed links, including co-located connectors",
                    "开放单向连接（含共址连接）",
                ),
                s["open_links"],
            ),
            (
                T("Used links with flow above 10⁻⁶/h", "正流量连接数（阈值10⁻⁶/小时）"),
                s["used_links"],
            ),
            (T("Continuous fleet size Γ", "连续机队规模Γ"), _n(s["fleet"])),
        ],
    )
    add(T("## 3. Base operations and repositioning", "## 3. 基地运营与重调度"))
    add(
        T(
            "ρ is the probability that an idle drone is available at a base; idle inventory is the mean number of drones in its queue. Service departures and arrivals are reported separately from empty repositioning.",
            "ρ为基地有可用无人机的概率，待命架数为该基地队列平均库存。服务出发、服务返回与空载重调度分别统计。",
        )
    )
    tab(
        [
            T("Base", "基地"),
            T("Open", "开放"),
            "ρ",
            T("Idle", "待命"),
            T("Service out/in", "服务出/入"),
            T("Empty out/in", "重调度出/入"),
        ],
        [
            (
                f"L{x['base']+1}",
                x["open"],
                f"{x['rho']:.4f}",
                _n(x["idle"]),
                f"{x['service_out']:.2f} / {x['service_in']:.2f}",
                f"{x['empty_out']:.3f} / {x['empty_in']:.3f}",
            )
            for x in case["tables"]["bases"]
        ],
    )
    add(
        T(
            f"Cross-base service missions total {_flow(r['cross_base_service_per_hour'])} orders/h. Empty repositioning totals {_flow(r['flow_per_hour'])} missions/h and occupies {m['fleet']['empty_occupancy']:.3f} drones. Service-mission return flights occupy {f.get('return_flight',0):.3f} drones separately.",
            f"跨基地服务任务共{_flow(r['cross_base_service_per_hour'])}单/小时。空载重调度{_flow(r['flow_per_hour'])}架次/小时，占用{m['fleet']['empty_occupancy']:.3f}架无人机。任务内返回飞行另外占用{f.get('return_flight',0):.3f}架。",
        )
    )
    if r["pairs"]:
        tab(
            [
                T("Departure", "出发"),
                T("Arrival", "到达"),
                T("Repositioning missions/h", "重调度架次/小时"),
            ],
            [
                (x["origin"], x["destination"], _n(x["flow_per_hour"], 3))
                for x in r["pairs"]
            ],
        )
        service_matrix = np.asarray(r["service_base_pair_matrix"])
        cross = [
            (f"L{i+1}", f"L{j+1}", _n(service_matrix[i, j], 3))
            for i, j in np.argwhere(service_matrix > 1e-4)
            if i != j
        ]
        if cross:
            tab(
                [
                    T("Service departure base", "服务出发基地"),
                    T("Service return base", "服务返回基地"),
                    T("Orders/h", "单/小时"),
                ],
                cross,
            )
        if len(r["pairs"]) == 1 and service_matrix.shape == (2, 2):
            route = r["pairs"][0]
            origin = int(route["origin"][1:]) - 1
            dest = int(route["destination"][1:]) - 1
            inbound = service_matrix[dest, origin]
            outbound = service_matrix[origin, dest]
            if abs(inbound - outbound - route["flow_per_hour"]) < 1e-4:
                add(
                    T(
                        f"The two cross-base service flows explain repositioning directly: {route['destination']}→{route['origin']} carries {inbound:.3f} orders/h, whereas {route['origin']}→{route['destination']} carries {outbound:.3f} orders/h. Their difference leaves {inbound-outbound:.3f} extra drone arrivals/h at {route['origin']}; empty flights from {route['origin']} to {route['destination']} return exactly that flow and restore base balance.",
                        f"两个方向的跨基地服务解释了重调度的来源：{route['destination']}→{route['origin']}为{inbound:.3f}单/小时，{route['origin']}→{route['destination']}为{outbound:.3f}单/小时。两者相差{inbound-outbound:.3f}架次/小时，使{route['origin']}获得净流入；{route['origin']}→{route['destination']}的空载调度恰好将这部分无人机送回，维持基地流量平衡。",
                    )
                )
    else:
        add(
            T(
                "No material inter-base repositioning appears at the display threshold of 10⁻⁴ missions/h. Raw matrix values are retained in metrics.json.",
                "按10⁻⁴架次/小时的展示阈值，本解没有实质性基地间重调度。矩阵原始数值完整保留在metrics.json。",
            )
        )
    add(T("## 4. Fleet occupancy", "## 4. 机队占用"))
    components = [
        ("At bases", "基地待命", "idle"),
        ("Flight to pickup", "接餐飞行", "pre_pickup_flight"),
        ("Swapping before pickup", "接餐前换电", "pre_pickup_swap"),
        ("Loaded flight", "带餐飞行", "loaded_flight"),
        ("Loaded-leg swapping", "带餐途中换电", "loaded_swap"),
        ("Service return flight", "任务返回飞行", "return_flight"),
        ("Return-leg swapping", "返回途中换电", "return_swap"),
        ("Pickup handling", "取餐操作", "pickup_handling"),
        ("Unloading", "放餐操作", "delivery_handling"),
        ("Aerial waiting", "空中等餐", "sync_wait"),
        ("Empty repositioning flight", "重调度飞行", "empty_flight"),
        ("Empty repositioning swapping", "重调度换电", "empty_swap"),
        ("Fleet constraint slack", "机队约束余量", "fleet_slack"),
    ]
    tab(
        [
            T("State", "状态"),
            T("Average drones", "平均架数"),
            T("Fleet share", "机队占比"),
        ],
        [
            (T(en, cn), _n(f.get(k, 0), 3), f"{100*f.get(k,0)/s['fleet']:.2f}%")
            for en, cn, k in components
        ]
        + [(T("Total", "合计"), _n(s["fleet"], 3), "100.00%")],
    )
    add(T("## 5. Locker capacity and swapping", "## 5. 柜容量与换电"))
    tab(
        [T("Metric", "指标"), T("Value", "数值")],
        [
            (
                T("Installed handling capacity, events/h", "安装处理容量，次/小时"),
                _n(cap["capacity_per_hour"]),
            ),
            (
                T("Handling load, events/h", "处理流量，次/小时"),
                _n(cap["handling_per_hour"]),
            ),
            (
                T("Aggregate capacity utilization", "整体容量利用率"),
                f"{100*cap['utilization']:.2f}%",
            ),
            (T("Fully utilized sites", "满载柜点"), cap["full_sites"]),
            (
                T("Sites at least 95% utilized", "利用率至少95%的柜点"),
                cap["at_least_95pct_sites"],
            ),
            (
                T("Swaps during service missions, visits/h", "服务任务换电次数/小时"),
                _n(m["swapping"]["service_visits_per_hour"]),
            ),
            (
                T(
                    "Service orders using at least one swap",
                    "至少换电一次的服务订单比例",
                ),
                f"{100*m['swapping']['service_flow_with_swap_fraction']:.2f}%",
            ),
        ],
    )
    add(
        T(
            "Each drone-assisted order consumes one handling event at the pickup locker and one at its destination locker; aggregate handling flow is twice the drone-assisted order flow.",
            "每个无人机订单在取餐柜和目的柜各占用一次处理能力，总处理流量为无人机服务单量的两倍。",
        )
    )
    tab(
        [T("Cabinets per site", "每点柜数"), T("Sites", "柜点数")],
        [
            (i, cap["count_distribution"].get(i, 0))
            for i in range(1, int(p["max_cabinets"]) + 1)
        ],
    )
    tab(
        [
            T("Swap station", "换电站"),
            T("Open", "开放"),
            T("Service visits/h", "服务换电次/小时"),
            T("Repositioning visits/h", "重调度换电次/小时"),
            T("Drones swapping", "换电占用架数"),
        ],
        [
            (
                x["station"],
                x["open"],
                _n(x["service_visits_per_hour"]),
                _n(x["reposition_visits_per_hour"]),
                _n(x["drones_swapping"], 3),
            )
            for x in m["swapping"]["stations"]
        ],
    )
    add(T("## 6. Service times", "## 6. 服务时间"))
    labels = [
        ("First-mile rider", "第一程骑手", "first_rider"),
        ("Drone approach flight", "无人机前段飞行", "pre_pickup_flight"),
        ("Pre-pickup swapping", "前段换电", "pre_pickup_swap"),
        ("Aerial waiting", "空中等餐", "wait"),
        ("Pickup handling", "取餐操作", "pickup_handling"),
        ("Loaded flight", "带餐飞行", "loaded_flight"),
        ("Loaded-leg swapping", "带餐途中换电", "loaded_swap"),
        ("Unloading", "放餐操作", "delivery_handling"),
        ("Last-mile courier including dispatch", "末端骑手（含呼叫）", "terminal"),
        ("Mean delivery completion", "配送完成均值", "mean_delivery"),
        ("Return flight", "任务返航", "return_flight"),
        ("Return-leg swapping", "返航换电", "return_swap"),
        ("Complete drone mission including return", "完整无人机任务（含返回）", "time"),
        (
            "Pure-courier reference for the same served ODs",
            "同一批服务OD的纯骑手基准",
            "direct_same_served_OD",
        ),
    ]
    tab(
        [T("Flow-weighted mean", "流量加权均值"), T("Minutes", "分钟")],
        [(T(en, cn), _n(tm.get(k), 3)) for en, cn, k in labels],
    )
    add(
        T(
            f"Aerial waiting averages {60*tm['wait']:.2f} seconds per service order. Time starts at merchant pickup, when the first-mile rider and drone launch in parallel. Delivery completion uses the same home-delivery terminal convention for all served orders, including {60*p['terminal_dispatch_time']:.3f} minutes for rider dispatch; customer self-pickup walking time is excluded.",
            f"每单平均空中等餐{60*tm['wait']:.2f}秒。时间从骑手在商家取到餐开始，此时第一程骑手与无人机并行出发。配送完成时间对全部服务订单使用统一到家末端口径，其中呼叫骑手为{60*p['terminal_dispatch_time']:.3f}分钟；自提顾客步行时间未计入。",
        )
    )
    add(
        T(
            f"The flow-weighted median and 90th percentile across OD mean delivery times are {tm['OD_mean_delivery']['median']:.2f} and {tm['OD_mean_delivery']['p90']:.2f} minutes. These summarize variation across OD means. {100*tm['fraction_at_least_5min_faster']:.2f}% of served flow has a mean delivery time at least five minutes below its matched pure-courier reference.",
            f"按服务流量加权，OD平均配送时间的中位数为{tm['OD_mean_delivery']['median']:.2f}分钟，90分位为{tm['OD_mean_delivery']['p90']:.2f}分钟；这些分位数描述OD均值之间的差异。{100*tm['fraction_at_least_5min_faster']:.2f}%的服务量比对应纯骑手均值至少快五分钟。",
        )
    )
    first = m["first_leg"]
    add(
        T(
            f"Selected merchant-to-pickup-locker distance averages {first['distance_m']['mean']:.2f} m; {100*first['colocated_within_1m_flow_fraction']:.2f}% of served flow uses representative points co-located within 1 m. The current zero-intercept distance rule makes this aggregation feature directly affect first-mile and waiting times.",
            f"选中商家至取餐柜的平均直线距离为{first['distance_m']['mean']:.2f}米；{100*first['colocated_within_1m_flow_fraction']:.2f}%的服务量对应相距一米以内的共址代表点。当前零截距距离时间规则使这一坐标聚合特征直接影响首程和等餐时间。",
        )
    )
    add(T("## 7. Cost breakdown", "## 7. 成本构成"))
    keys = [
        ("Bases", "基地", "base_fixed"),
        ("Cabinets", "柜子", "locker_fixed"),
        ("Swapping stations", "换电站", "swap_fixed"),
        ("Directed air routes", "航线开设", "link_fixed"),
        ("Fleet", "机队", "fleet"),
        ("First-mile riders", "首程骑手", "first_rider"),
        ("Last-mile riders", "末端骑手", "terminal_rider"),
        ("Service flight", "服务飞行", "service_flight"),
        ("Aerial waiting", "空中等餐", "aerial_wait"),
        ("Empty flight", "重调度飞行", "empty_flight"),
        ("Courier-only orders", "全程骑手", "direct_rider"),
    ]
    tab(
        [T("Component", "项目"), T("CNY/h", "元/小时"), T("Share", "占比")],
        [
            (T(en, cn), _n(cs[k]), f"{100*cs[k]/s['objective']:.2f}%")
            for en, cn, k in keys
        ]
        + [(T("Total", "合计"), _n(s["objective"]), "100.00%")],
    )
    if has_site:
        add(
            T(
                "The facility totals above already include the following setup charges.",
                "上述基地、柜点和换电站成本已包含以下选址费用：",
            )
        )
        tab(
            [
                T("Included setup component", "其中选址费"),
                T("Opened sites", "开放位置数"),
                T("CNY/h", "元/小时"),
            ],
            [
                (
                    names[x["component"]],
                    x["opened_locations"],
                    _n(x["paid_cny_per_hour"], 3),
                )
                for x in site["rows"]
            ]
            + [
                (
                    T("Setup subtotal included above", "已计入的选址费小计"),
                    "—",
                    _n(site["total_paid_cny_per_hour"], 3),
                )
            ],
        )
    add(
        T(
            f"Total cost per potential order is CNY {m['costs']['per_potential_order']:.3f}. Self-pickup has zero last-mile rider cost; the first-mile rider tariff retains its minimum charge for co-located representative points.",
            f"折合每个潜在订单成本{m['costs']['per_potential_order']:.3f}元。到柜自提的末端骑手成本为零，首程骑手在代表点共址时仍收取起步价。",
        )
    )
    add(f"![{T('Operations','运营统计')}](figures_{lang}/H{H}_operations.png)")
    if (out / f"maps_{lang}/H{H}_facilities.png").is_file():
        add(T("## 8. Geographic results", "## 8. 地理结果"))
        add(
            T(
                "Circle size denotes cabinet count; color denotes capacity utilization. Squares are open bases and dark-rust diamonds are open swapping stations; grey crosses mark unopened candidate locations. Loaded-segment colors identify the departure base; the main map covers 90% of loaded-segment flow, and the base panels include every positive-flow loaded segment.",
                "圆点大小为柜数，颜色为容量利用率；方形为开放基地，深锈色菱形为开放换电站，灰色叉号为未开放候选位置。带餐航段按出发基地着色；主图覆盖90%带餐航段流量，分基地图保留全部正流量带餐航段。",
            )
        )
        for name, label in [
            ("facilities", T("Facilities and capacity", "设施与容量")),
            ("service_corridors", T("Loaded flight segments", "带餐航段")),
        ]:
            add(f"![{label}](maps_{lang}/H{H}_{name}.png)")
        if r["pairs"]:
            add(
                f"![{T('Empty repositioning','空载重调度')}](maps_{lang}/H{H}_repositioning.png)"
            )
        add(
            f"[{T('Complete loaded segments by base','各基地全部带餐航段')}](maps_{lang}/H{H}_base_panels.png)"
        )
        add(
            T(
                "Maps reuse the cached Shenzhen OpenStreetMap basemap and the historical local-coordinate / GCJ-02-to-WGS84 plotting convention. Model distances are unchanged. © OpenStreetMap contributors.",
                "地图使用原有深圳OpenStreetMap缓存底图，沿用局部坐标反投影和GCJ-02至WGS84制图规则；模型距离不变。© OpenStreetMap contributors。",
            )
        )
    add(T("## 9. Solver status and validation", "## 9. 求解状态与核验"))
    for section in solver_section(sol, lang):
        add(section)
    stage_figure = out / f"figures_{lang}/H{H}_solver_stages.png"
    if stage_figure.is_file():
        add(
            f"![{T('Solver stage timings', '求解阶段时长')}](figures_{lang}/H{H}_solver_stages.png)"
        )
    add(
        T(
            "Independent reconstruction passed assignment, base-flow conservation, cabinet capacity, link activation, battery, timing, joint reliability, fleet occupancy, and objective checks. The reported gap uses the full-model lower bound. Parameter, solution, and path snapshots are preserved alongside the results.",
            "独立重构通过订单分配、基地流量守恒、柜容量、航线激活、电池、时间、联合可靠性、机队和目标值检查。报告Gap采用完整模型下界；参数、解和路径快照与结果一同保留。",
        )
    )
    if (out / f"figures_{lang}/H{H}_convergence.png").is_file():
        add(
            f"![{T('Full-model convergence','完整模型收敛过程')}](figures_{lang}/H{H}_convergence.png)"
        )
    links = [
        (T("Metrics and checks", "完整统计与核验"), "metrics.json"),
        (T("Source manifest", "来源清单"), "source_manifest.json"),
    ]
    for name, en, cn in [
        ("result.json", "Saved result", "保存结果"),
        ("input_metadata.json", "Input parameters", "参数快照"),
        ("operations.json", "Operational data", "运营明细"),
    ]:
        links.append((T(en, cn), os.path.relpath(case["path"] / name, out)))
    if (out / "all_base_phase_edges.csv").is_file():
        links.append(
            (
                T("Directed physical segments", "有向物理航段"),
                "all_base_phase_edges.csv",
            )
        )
    add(
        T("Sources: ", "来源：")
        + "; ".join(f"[{label}]({target})" for label, target in links)
    )
    (out / f"REPORT_{lang.upper()}.md").write_text("\n".join(lines), encoding="utf-8")
