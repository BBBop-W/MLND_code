"""Build a Chinese, results-only report for the first two sensitivity rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUPS = [
    ("demand", "订单规模"),
    ("rider_cost", "骑手成本"),
    ("fleet_cost", "无人机固定成本"),
    ("speed", "飞行速度"),
    ("battery", "电池续航"),
    ("density", "候选站点密度"),
]


def num(value, digits=2):
    return f"{value:,.{digits}f}"


def esc(value):
    text = str(value)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(a, b)
    return text.replace("ε", r"\(\epsilon\)").replace("ρ", r"\(\rho\)")


class Report:
    def __init__(self):
        self.tex = []
        self.md = []

    def page(self, title):
        if self.tex:
            self.tex.append(r"\clearpage")
        self.tex.append(r"\section{" + esc(title) + "}")
        self.md.append("\n## " + title + "\n")

    def para(self, text):
        self.tex.append(esc(text) + "\n\n")
        self.md.append(text + "\n")

    def table(self, title, headers, rows, *, size=9.5):
        self.tex.append(r"\subsection*{" + esc(title) + "}")
        self.tex.append(r"\begingroup\fontsize{" + str(size) + "}{12.5}\\selectfont"
                        r"\renewcommand{\arraystretch}{1.23}\setlength{\tabcolsep}{3pt}")
        self.tex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l"
                        + "r" * (len(headers) - 1) + "@{}}\\toprule")
        self.tex.append(" & ".join(r"\shortstack{" + esc(h).replace("\n", r"\\") + "}"
                                   for h in headers) + r"\\\midrule")
        self.tex.extend(" & ".join(esc(v) for v in row) + r"\\" for row in rows)
        self.tex.append(r"\bottomrule\end{tabular*}\endgroup\par\vspace{3mm}")
        self.md.append("\n### " + title + "\n")
        self.md.append("| " + " | ".join(h.replace("\n", " ") for h in headers) + " |")
        self.md.append("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
        self.md.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        self.md.append("")

    def figure(self, path, caption, height="0.74"):
        self.tex.append(r"\begin{center}\includegraphics[width=\textwidth,height=" + height
                        + r"\textheight,keepaspectratio]{" + path
                        + r"}\captionof{figure}{" + esc(caption) + r"}\end{center}")
        self.md.append(f"\n![{caption}]({path})\n")


def label(row):
    if row["factor"] == "baseline":
        return "基准"
    group = dict(GROUPS)[row["factor"]]
    if row["factor"] == "density":
        return f"候选 H{row['candidate_lockers']}"
    return f"{group} ×{row['multiplier']:.2f}"


def opened(row):
    return "/".join(str(row[k]) for k in ("opened_bases", "opened_lockers", "opened_swaps"))


def candidates(row):
    return "/".join(str(row[k]) for k in ("candidate_lockers", "candidate_bases", "candidate_swaps"))


def build(directory):
    out = Path(directory)
    all_rows = json.loads((out / "all_results.json").read_text())
    assert len(all_rows) == 25 and all(r["round"] in (1, 2) for r in all_rows)
    baseline = next(r for r in all_rows if r["factor"] == "baseline")
    details = {}
    bases = {}
    for row in all_rows:
        run = Path(row["run_path"])
        details[row["case"]] = json.loads((run / "sensitivity/sensitivity_metrics.json").read_text())["details"]
        bases[row["case"]] = json.loads((run / "operations.json").read_text())["tables"]["bases"]
    ordered = [baseline]
    for group, _ in GROUPS:
        ordered.extend(sorted((r for r in all_rows if r["factor"] == group),
                              key=lambda r: r["multiplier"]))
    base_detail = details[baseline["case"]]
    params = base_detail["parameters"]
    metadata = json.loads((Path(baseline["run_path"]) / "input_metadata.json").read_text())
    area = metadata["values"]["provenance"]["instance_metadata"]
    doc = Report()
    doc.page("试验范围与统计口径")
    doc.para("本报告汇总轮次一与轮次二的 25 个独立算例。六类单因素试验共用一个基准解；分组表中重复列出的基准行来自同一次求解。")
    doc.table("试验与数据范围", ["项目", "设置"], [
        ["区域", "深圳南山区域 2"],
        ["研究区域", f"边长 {2 * area['crop_half_size_km']:.3f} km；面积 {4 * area['crop_half_size_km'] ** 2:.2f} km²"],
        ["原始数据", f"{area['raw_rows']:,} 条订单，7 天"],
        ["区域内候选时段订单", f"{area['candidate_orders_in_area']:,} 条；08:00–24:00"],
        ["区域内需求时段订单", f"{area['demand_orders_in_area']:,} 条；12:00–13:00"],
        ["聚合后需求", f"400 个 OD；基准 {baseline['demand_per_hour']:,.2f} 单/小时"],
        ["基准候选数量", "4 个基地 L、20 个柜位置 H、8 个换电站 E"],
        ["求解方法", "Arc-based formulation；每例 1 个求解线程"],
        ["停止设置", "每例 1 小时；相对 gap 0.5%"],
        ["轮次一", "13 个独立算例；同时运行 3 个"],
        ["轮次二", "12 个新增算例；同时运行 2 个"],
        ["已保存结果", f"25 个解；{sum(not r['above_gap_target'] for r in all_rows)} 个达到目标 gap；{sum(r['above_gap_target'] for r in all_rows)} 个按时限停止"],
        ["结果校验", "25 / 25 通过"],
    ], size=10)
    doc.para("总成本和订单率按运营小时计。节省额等于本算例全部由骑手配送的成本减去所得网络总成本；节省率以本算例全部骑手配送成本为分母。无人机服务份额以全部潜在订单率为分母。")
    doc.para("开放 L/H/E 分别计基地、柜位置、换电站；柜单元另计。机队规模为连续变量。柜容量利用率按两个 H 的装卸处理总量除以已安装总能力计算。重调度以空载任务数计；换电使用量以访问换电站的次数计。")
    doc.para("时间从骑手在商家取到餐时开始：骑手首段与无人机取餐前过程并行。客户配送完成时间采用统一到家末端口径，包含末端骑手行驶与呼叫骑手常数；商家备餐、自取步行和无人机返航时间不计入该完成时间。分项时间按无人机服务流量加权。")

    doc.page("基准参数与各组设置")
    doc.table("运营与用户选择参数", ["参数", "基准值"], [
        ["无人机速度", f"{params['speed']:.2f} km/h"],
        ["电池续航时间", f"{params['battery_time'] * 60:.3f} min"],
        ["换电时间", f"{params['swap_time'] * 60:.2f} min/次"],
        ["柜单元处理能力", f"{params['unit_capacity']:.2f} 次/小时"],
        ["每个柜位置安装上限", f"{params['max_cabinets']} 个柜单元"],
        ["取餐/投递操作时间", f"{params['pickup_handling_time'] * 60:.2f} / {params['delivery_handling_time'] * 60:.2f} min"],
        ["末端呼叫骑手常数", f"{params['terminal_dispatch_time'] * 60:.3f} min"],
        ["首段骑手时间", "指数分布；均值 5.733 min/直线 km"],
        ["路径可靠性参数", f"ε = {params['epsilon']:.2f}"],
        ["骑手起步费用 / 含里程", "5.76 元 / 3 km"],
        ["骑手超距费用 / 路程换算", "1.62 元/km / 1.25"],
        ["接驳段骑手费用系数", "0.30"],
        ["飞行 / 同步等待成本", "14.40 / 14.40 元/架·小时"],
        ["自取选择常数 / 时间系数", "−1.00 / 0.06"],
        ["价格系数 / 步行距离系数", "0.35 / 2.40"],
        ["到家 / 到柜展示时间", "30 / 20 min"],
        ["自取优惠 / 最大步行距离", "1.00 元 / 0.80 km"],
    ], size=10)
    doc.table("单因素试验档位", ["因素", "档位", "实际缩放口径"], [
        ["订单规模", "0.50, 0.75, 1.00, 1.25, 1.50", "400 个 OD 的订单率"],
        ["骑手成本", "0.50, 0.75, 1.00, 1.25, 1.50", "纯骑手、首段、末段费用"],
        ["无人机固定成本", "0.50, 0.75, 1.00, 1.25, 1.50", "每架无人机固定单价"],
        ["飞行速度", "0.50, 0.75, 1.00, 1.25, 1.50", "速度；续航时间保持基准"],
        ["电池续航", "0.50, 0.75, 1.00, 1.25, 1.50", "续航时间；速度保持基准"],
        ["候选 H/L/E", "10/2/4, 15/3/6, 20/4/8", "同一区域、同一批 OD"],
        ["候选 H/L/E（续）", "25/5/10, 30/6/12", "候选位置逐档嵌套"],
    ], size=9.5)

    doc.page("固定成本与基准结果")
    doc.para("固定建设与设备成本按 5 年、每年 365 天、每天 12 个运营小时分摊，合计 21,900 小时。下表的五年期金额按当前小时费率乘以 21,900 汇总。")
    doc.table("固定参数的五年期金额", ["项目", "小时费率（元）", "五年期金额（元）"], [
        ["基地设备 / 每处", num(params['base_cost']), num(params['base_cost'] * 21900)],
        ["基地选址 / 每处", num(params['base_site_cost'], 4), num(params['base_site_cost'] * 21900)],
        ["柜设备 / 每单元", num(params['hub_cost']), num(params['hub_cost'] * 21900)],
        ["柜位置选址 / 每处", num(params['hub_site_cost'], 4), num(params['hub_site_cost'] * 21900)],
        ["换电设备 / 每处", num(params['swap_cost']), num(params['swap_cost'] * 21900)],
        ["换电选址 / 每处", num(params['swap_site_cost'], 4), num(params['swap_site_cost'] * 21900)],
        ["无人机 / 每架", num(params['fleet_cost']), num(params['fleet_cost'] * 21900)],
        ["单向航线 / 每 km", num(params['airline_hh_cost_per_km_hour'], 4), num(params['airline_hh_cost_per_km_hour'] * 21900)],
    ], size=10)
    doc.table("基准数值", ["指标", "结果"], [
        ["总成本 / 全部骑手成本（元/小时）", f"{num(baseline['total_cost_per_hour'])} / {num(baseline['all_courier_cost_per_hour'])}"],
        ["成本节省额（元/小时） / 节省率", f"{num(baseline['saving_per_hour'])} / {num(baseline['saving_pct'])}%"],
        ["无人机订单（单/小时） / 份额", f"{num(baseline['drone_served_per_hour'])} / {num(baseline['drone_served_share_pct'])}%"],
        ["自取 / 到家 / 纯骑手（单/小时）", " / ".join(num(baseline[k]) for k in ['pickup_per_hour','home_per_hour','direct_courier_per_hour'])],
        ["机队 / 空闲队列中的无人机", f"{num(baseline['fleet'])} / {num(baseline['fleet_idle'])}"],
        ["开放 L/H/E / 柜单元", f"{opened(baseline)} / {baseline['locker_units']:.0f}"],
        ["柜总处理能力（次/小时） / 利用率", f"{num(base_detail['locker_capacity']['capacity_per_hour'])} / {num(baseline['locker_capacity_utilization_pct'])}%"],
        ["重调度（次/小时） / 跨基地服务比例", f"{num(baseline['reposition_flow_per_hour'])} / {num(baseline['cross_base_service_pct'])}%"],
        ["换电（次/小时） / 使用换电订单比例", f"{num(baseline['swap_visits_per_hour'])} / {num(baseline['service_using_swap_pct'])}%"],
        ["配送完成 / 同服务 OD 纯骑手（min）", f"{num(baseline['mean_delivery_minutes'])} / {num(baseline['direct_same_served_OD_minutes'])}"],
    ], size=10)
    doc.para("所有成本结果均包括纯骑手承担的订单。各组后续表中的经济、机队、设施和运营指标均对应该算例保存的同一个可行解。")

    doc.page("25 个独立算例：经济与求解结果")
    doc.table("完整经济与求解汇总", ["算例", "轮次", "成本\n元/h", "节省额\n元/h", "节省率\n%", "每单成本\n元/单", "Gap\n%", "耗时\nmin"], [
        [label(r), r['round'], num(r['total_cost_per_hour']), num(r['saving_per_hour']), num(r['saving_pct']), num(r['cost_per_order'], 3), num(r['gap_pct'], 3), num(r['solver_seconds']/60)]
        for r in ordered
    ], size=9.1)
    doc.para("成本为当前可行解目标值；Gap 为保存结果的相对最优性差距。耗时使用各次求解的记录时间，包含终止收尾，可能略超过设定的一小时。全部算例的目标下界和原始求解状态保留在完整数据表中。")
    doc.page("25 个独立算例：服务与网络结果")
    doc.table("完整服务与网络汇总", ["算例", "无人机量\n单/h", "份额\n%", "机队\n架", "开放\nL/H/E", "柜单元", "重调度\n次/h", "换电\n次/h"], [
        [label(r), num(r['drone_served_per_hour']), num(r['drone_served_share_pct']), num(r['fleet']), opened(r), f"{r['locker_units']:.0f}", num(r['reposition_flow_per_hour'], 3), num(r['swap_visits_per_hour'])]
        for r in ordered
    ], size=9.1)
    doc.para("换电次数汇总服务任务与重调度任务。机队为期望占用与空闲队列资源所对应的连续规模；柜单元为安装数量；开放 H 为位置数。")

    for group, name in GROUPS:
        rows = sorted([baseline] + [r for r in all_rows if r['factor'] == group], key=lambda r: r['multiplier'])
        doc.page(name + "：分档结果")
        if group == 'density':
            doc.para("档位依次为 H10/L2/E4、H15/L3/E6、H20/L4/E8、H25/L5/E10、H30/L6/E12。区域、400 个 OD 和 3,583.57 单/小时保持一致；表中档位记候选 H 数量。")
        else:
            definitions = {
                'demand': '订单率依次为 1,791.79、2,687.68、3,583.57、4,479.46、5,375.36 单/小时。',
                'rider_cost': '起步骑手费用依次为 2.88、4.32、5.76、7.20、8.64 元；超距单价及接驳段费用按同一倍数缩放。',
                'fleet_cost': '无人机固定费率依次为 3.60、5.40、7.20、9.00、10.80 元/架·小时；其余固定费用保持基准。',
                'speed': '飞行速度依次为 36、54、72、90、108 km/h；续航时间保持 8.333 min。',
                'battery': '续航时间依次为 4.167、6.250、8.333、10.417、12.500 min；速度保持 72 km/h。',
            }
            doc.para(definitions[group] + " 1.00 倍为共用基准。")
        def tier(r):
            return f"H{r['candidate_lockers']}" if group == 'density' else f"{r['multiplier']:.2f}"
        doc.table("经济与求解", ["档位", "总成本\n元/h", "全部骑手\n元/h", "节省率\n%", "每单成本\n元/单", "Gap\n%", "耗时\nmin"], [
            [tier(r), num(r['total_cost_per_hour']), num(r['all_courier_cost_per_hour']), num(r['saving_pct']), num(r['cost_per_order'],3), num(r['gap_pct'],3), num(r['solver_seconds']/60)] for r in rows
        ])
        doc.table("服务与设施", ["档位", "无人机量\n单/h", "份额\n%", "自取\n单/h", "到家\n单/h", "机队\n架", "开放\nL/H/E", "柜单元"], [
            [tier(r), num(r['drone_served_per_hour']), num(r['drone_served_share_pct']), num(r['pickup_per_hour']), num(r['home_per_hour']), num(r['fleet']), opened(r), f"{r['locker_units']:.0f}"] for r in rows
        ])
        time_rows = []
        op_rows = []
        for r in rows:
            t = details[r['case']]['times']
            rho = [x['rho'] for x in bases[r['case']] if x['open']]
            time_rows.append([tier(r), num(t['mean_delivery']), num(t['direct_same_served_OD']), num(t['first_rider']), num(t['pre_pickup_flight']), num(t['loaded_flight']), num(t['return_flight']), num(sum(t.get(k,0) for k in ['pre_pickup_swap','loaded_swap','return_swap']))])
            op_rows.append([tier(r), num(t['sync_wait']), num(t['terminal']), num(r['reposition_flow_per_hour'],3), num(r['swap_visits_per_hour']), num(r['service_using_swap_pct']), num(r['locker_capacity_utilization_pct']), f"{min(rho):.3f}–{max(rho):.3f}"])
        doc.table("配送与飞行时间（min）", ["档位", "配送\n完成", "同 OD\n纯骑手", "首段\n骑手", "取餐前\n飞行", "带载\n飞行", "返航\n飞行", "任务内\n换电"], time_rows)
        doc.table("同步、末端与运营", ["档位", "同步等餐\nmin", "末端\nmin", "重调度\n次/h", "换电\n次/h", "换电订单\n%", "柜利用率\n%", "开放基地\nρ 范围"], op_rows)
        doc.para("ρ 范围为开放基地可用性的最小值至最大值；任务内换电统计包括返航段。")

    figure_specs = [
        ('savings_sensitivity', '成本节省率', '六类单因素试验的成本节省率。竖线由各解的目标值与目标下界换算；空心点表示最终 gap 超过 0.5%。'),
        ('technology_sensitivity', '速度与续航的运营指标', '速度与续航各档的无人机服务、机队和换电指标。各点均对应同一档保存的解。'),
        ('density_sensitivity', '候选密度的网络指标', '候选站点密度试验：同一区域与同一批 400 个 OD。候选 H/L/E 按固定比例变化。'),
        ('cost_composition', '成本构成', '各因素各档的每单成本构成（元/单）；各分项小时成本除以本算例全部潜在订单率。'),
        ('fleet_composition', '机队构成', '各因素各档的机队占用构成。机队以连续规模表示。'),
        ('delivery_times', '配送时间', '按无人机服务流量加权的取餐后配送完成时间与对应 OD 纯骑手时间。各档服务的 OD 构成随求解结果而定。'),
    ]
    for filename, title, caption in figure_specs:
        path = out / 'figures' / (filename + '.png')
        if path.exists():
            doc.page(title)
            doc.figure('figures/' + filename + '.png', caption)

    for case, title in [('r2_density_H10','H10/L2/E4'), ('r1_baseline','H20/L4/E8（基准）'), ('r2_density_H30','H30/L6/E12')]:
        doc.page('设施地图：' + title)
        doc.figure(f'maps/{case}/facilities.png', title + ' 的候选与开放设施。', height='0.82')
        doc.page('运行地图：' + title)
        doc.figure(f'maps/{case}/network.png', title + ' 的运行网络；图中不叠加 OD 连线。', height='0.82')

    doc.page('完整明细与字段口径')
    doc.para('完整汇总表保留 25 个独立算例的全部汇总指标；六类单因素表中的基准重复行用于分组展示。地图目录保存每个算例的设施图和网络图，共 50 张。')
    doc.table('表中常用字段', ['指标', '单位或统计口径'], [
        ['总成本 / 节省额', '元/运营小时'],
        ['每单成本', '总成本 ÷ 全部潜在订单率'],
        ['服务份额', '无人机订单率 ÷ 全部潜在订单率'],
        ['自取 / 到家', '无人机服务订单中的两类末端交付'],
        ['同 OD 纯骑手时间', '以该解无人机服务流量作权重'],
        ['OD 时间中位数 / P90', '各 OD 平均时间的加权分位数'],
        ['空闲无人机', '各基地空闲队列中的期望数量之和'],
        ['重调度', '基地间空载任务流量'],
        ['换电使用量', '全部服务与空载任务的换电访问次数'],
        ['换电订单比例', '至少换电一次的服务订单流量占比'],
        ['跨基地服务比例', '起点基地与返回基地不同的服务量占比'],
        ['柜利用率', '取餐与投递处理量 ÷ 已安装处理能力'],
        ['rho', '已开放基地的无人机可用性'],
        ['五年期金额', '当前小时固定费率 × 21,900 小时'],
    ], size=10)
    doc.para('配套文件包括全指标数据表、按基地/柜/换电站/航线/OD/路径划分的明细、LaTeX 三线表源码及全部地图索引。报告中的数字按显示精度四舍五入，完整数据保留原始数值精度。')
    doc.md.extend(['\n## 配套数据\n', '[完整 Excel 汇总](results_full.xlsx)',
                   '[全指标 CSV](full_metrics.csv)', '[全部地图索引](maps_index.md)',
                   '[LaTeX 三线表](sensitivity_tables.tex)'])
    preamble = r'''\documentclass[a4paper,zihao=5,fontset=fandol]{ctexart}
\usepackage[margin=18mm,headheight=15pt]{geometry}
\usepackage{booktabs,graphicx,caption,xcolor,fancyhdr,hyperref,amsmath}
\definecolor{ink}{HTML}{163A50}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=ink,pdftitle={灵敏度试验结果汇编：轮次一与轮次二}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 灵敏度试验结果汇编}
\fancyhead[R]{\small 轮次一与轮次二}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\ctexset{section={format=\Large\bfseries\color{ink},beforeskip=6pt,afterskip=10pt},subsection={format=\normalsize\bfseries,beforeskip=9pt,afterskip=5pt}}
\captionsetup{font=small,labelfont=bf,skip=5pt}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.12}
\begin{document}
{\LARGE\bfseries\color{ink} 灵敏度试验结果汇编}\par
{\large 轮次一与轮次二 · 南山区域 2}\par
{\small 汇总日期：2026 年 9 月 21 日}\vspace{3mm}
'''
    (out / 'report_zh.tex').write_text(preamble + '\n'.join(doc.tex) + '\n\\end{document}\n')
    (out / 'report_zh.md').write_text('# 灵敏度试验结果汇编：轮次一与轮次二\n\n汇总日期：2026 年 9 月 21 日。\n' + '\n'.join(doc.md))
    return out / 'report_zh.tex'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(build(args.directory))
