"""Create the results-only supplement for the third sensitivity round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .complete_report import Report, num, opened


GROUPS = [
    ("demand", "订单规模", "Demand multiplier"),
    ("speed", "无人机飞行速度", "Flight-speed multiplier"),
    ("max_cabinets", "每个 H 的柜单元上限", "Maximum cabinets per H"),
    ("walk_sensitivity", "步行距离敏感系数", "Walking-distance coefficient multiplier"),
]


def tier(row):
    if row["factor"] == "max_cabinets":
        return str(row["max_cabinets"])
    return f"{row['multiplier']:.2f}"


def label(row):
    if row["factor"] == "max_cabinets":
        return f"柜上限 {row['max_cabinets']}"
    prefix = {"demand": "订单", "speed": "速度", "walk_sensitivity": "步行系数"}[row["factor"]]
    return f"{prefix} ×{row['multiplier']:.2f}"


def make_tables(out, rows):
    """Write four copy-ready panels containing the 15 new runs only."""
    blocks = []
    for factor, _, title in GROUPS:
        subset = [r for r in rows if r["factor"] == factor]
        columns = "l" + "r" * len(subset)
        specs = [
            ("Cost per order, CNY", lambda r: num(r["cost_per_order"])),
            (r"Cost saving, \%", lambda r: num(r["saving_pct"])),
            (r"Drone-served share, \%", lambda r: num(r["drone_served_share_pct"])),
            ("Fleet size", lambda r: num(r["fleet"], 1)),
            ("Opened L/H/E", opened),
            ("Installed cabinet units", lambda r: str(round(r["locker_units"]))),
            (r"MIP gap, \%", lambda r: num(r["gap_pct"]) + (r"$^{\dagger}$" if r["above_gap_target"] else "")),
            ("Time, min", lambda r: num(r["solver_seconds"] / 60, 1)),
        ]
        caption = {
            "demand": "Additional sensitivity results: order volume",
            "speed": "Additional sensitivity results: drone speed",
            "max_cabinets": "Sensitivity to the maximum number of cabinets per locker location",
            "walk_sensitivity": "Sensitivity to walking-distance preferences",
        }[factor]
        note = {
            "demand": "The multiplier applies to the order rates of the same 400 OD pairs. The unscaled demand is 3,583.57 orders/hour.",
            "speed": "The unscaled speed is 72 km/h. Battery endurance remains 8.333 min.",
            "max_cabinets": "Columns give the maximum number of cabinet units at each H location. Each unit handles 30.4 events/hour. Candidate locations remain unchanged.",
            "walk_sensitivity": "The unscaled walking-distance coefficient is 2.4. Columns therefore use coefficients 1.2, 1.8, 3.0, and 3.6. The 800 m cutoff remains unchanged.",
        }[factor]
        lines = [
            r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}",
            r"\label{tab:sensitivity_round3_" + factor + "}", r"\begingroup",
            r"\small\setlength{\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.17}",
            r"\begin{tabular}{@{}" + columns + r"@{}}\toprule",
            r" & \multicolumn{" + str(len(subset)) + "}{c}{" + title + r"}\\",
            r"\cmidrule(l){2-" + str(len(subset) + 1) + "}",
            "Metric & " + " & ".join(tier(r) for r in subset) + r"\\\midrule",
        ]
        for name, getter in specs:
            lines.append(name + " & " + " & ".join(getter(r) for r in subset) + r"\\")
        lines.extend([
            r"\bottomrule\end{tabular}", r"\par\vspace{5pt}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"\textit{Notes.} " + note,
            "All cells are actual solution levels. Cost per order uses total potential demand; savings use the scenario-specific all-courier cost. Fleet size is continuous. L/H/E denotes bases, locker locations, and swap stations.",
            (r"$\dagger$ marks the three runs reaching the one-hour limit above the 0.5\% gap target." if factor == "demand" else r"All runs in this table meet the 0.5\% gap target."),
            r"\end{minipage}\endgroup\end{table}",
        ])
        blocks.append("\n".join(lines))
    (out / "sensitivity_tables.tex").write_text(
        "% Requires \\usepackage{booktabs}. Round 3 only: 15 new runs; no repeated baseline.\n"
        + "\n\n".join(blocks) + "\n"
    )
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\parindent}{0pt}
\begin{document}
"""
    (out / "tables_preview.tex").write_text(
        preamble + "\n\n".join(blocks[:2]) + "\n\\clearpage\n"
        + "\n\n".join(blocks[2:]) + "\n\\end{document}\n"
    )


def build(directory):
    out = Path(directory)
    rows = json.loads((out / "all_results.json").read_text())
    assert len(rows) == 15 and all(r["round"] == 3 for r in rows)
    assert all(r["validation_passed"] for r in rows)
    rows = [r for factor, _, _ in GROUPS for r in sorted(rows, key=lambda r: r["multiplier"]) if r["factor"] == factor]
    details, bases = {}, {}
    for row in rows:
        run = Path(row["run_path"])
        details[row["case"]] = json.loads((run / "sensitivity/sensitivity_metrics.json").read_text())["details"]
        bases[row["case"]] = json.loads((run / "operations.json").read_text())["tables"]["bases"]
    doc = Report()
    doc.page("新增试验范围")
    doc.para("本增补收录轮次三的全部 15 个新增算例。与轮次一、二已交付的 25 个独立算例合计，共 40 个独立算例；本文件的结果表仅列新增内容，不重复既有结果或基准行。")
    doc.table("轮次三的设置变化", ["因素", "新增档位", "实际取值"], [
        ["订单规模", "0.20、0.30、0.40 倍", "716.71、1,075.07、1,433.43 单/h"],
        ["无人机速度", "0.60、0.70、0.80、0.90 倍", "43.2、50.4、57.6、64.8 km/h"],
        ["每个 H 的柜单元上限", "3、4、6、7", "基准上限为 5 个柜单元"],
        ["步行距离敏感系数", "0.50、0.75、1.25、1.50 倍", "1.2、1.8、3.0、3.6"],
    ], size=9.4)
    doc.para("各组均为单因素试验。候选位置保持 H20/L4/E8，同为深圳南山区域 2 的 400 个 OD。订单缩放以 3,583.57 单/小时为 1.00 倍；速度缩放以 72 km/h 为 1.00 倍，续航时间保持 8.333 min。每个柜单元处理能力为 30.4 次/小时。步行距离敏感系数以 2.4 为基准，自取距离达到 800 m 及以上的截断设置保持不变。")
    doc.table("求解与保存", ["项目", "记录"], [
        ["求解方法", "Arc-based formulation"],
        ["并行设置", "同时运行 2 个算例；每例 1 个求解线程"],
        ["停止设置", "每例 1 小时；相对 gap 0.5%"],
        ["轮次三结果", "15 个保存解；12 个达到目标 gap；3 个达到时限"],
        ["时限算例", "订单规模 0.20、0.30、0.40 倍"],
        ["结果校验", "15 / 15 通过"],
    ], size=10)
    doc.para("成本按运营小时计；每单成本以全部潜在订单率为分母。节省率以各算例全部由骑手配送的成本为分母。开放 L/H/E 分别表示基地、柜位置、换电站；柜单元另计。机队规模为连续变量。")
    doc.para("配送时间从骑手在商家取餐时开始，采用共同的到家末端时间口径，包含末端骑手行驶和呼叫骑手常数；不含备餐、自取步行及无人机返航时间。配送和飞行分项以本解无人机服务流量加权；对应纯骑手时间使用相同服务 OD 和权重。")

    doc.page("15 个新增算例：经济与求解结果")
    doc.table("经济与求解汇总", ["算例", "成本\n元/h", "节省额\n元/h", "节省率\n%", "每单成本\n元/单", "Gap\n%", "耗时\nmin"], [
        [label(r), num(r["total_cost_per_hour"]), num(r["saving_per_hour"]), num(r["saving_pct"]), num(r["cost_per_order"], 3), num(r["gap_pct"], 3), num(r["solver_seconds"] / 60)] for r in rows
    ], size=9.5)
    doc.para("成本为所保存可行解的目标值。Gap 为求解器记录的相对最优性差距；记录时长包含终止收尾，可能略超过一小时。原始目标下界和求解状态保存在全指标数据中。")
    doc.page("15 个新增算例：服务与网络结果")
    doc.table("服务与网络汇总", ["算例", "无人机量\n单/h", "份额\n%", "机队\n架", "开放\nL/H/E", "柜单元", "重调度\n次/h", "换电\n次/h"], [
        [label(r), num(r["drone_served_per_hour"]), num(r["drone_served_share_pct"]), num(r["fleet"]), opened(r), str(round(r["locker_units"])), num(r["reposition_flow_per_hour"], 3), num(r["swap_visits_per_hour"])] for r in rows
    ], size=9.4)
    doc.para("重调度指基地间空载任务流量；换电次数汇总服务与空载任务中对换电站的访问。开放 H 的数量为柜位置数，安装柜单元数量为该网络实际配置的总数。")

    for factor, title, _ in GROUPS:
        subset = [r for r in rows if r["factor"] == factor]
        doc.page(title + "：新增分档结果")
        definitions = {
            "demand": "档位为全部 400 个 OD 订单率的倍率；订单率依次为 716.71、1,075.07、1,433.43 单/小时。",
            "speed": "档位为飞行速度倍率；实际速度依次为 43.2、50.4、57.6、64.8 km/h。",
            "max_cabinets": "档位直接表示每个 H 位置可安装柜单元的上限，依次为 3、4、6、7；每单元处理能力保持 30.4 次/小时。",
            "walk_sensitivity": "档位为步行距离敏感系数的倍率；实际系数依次为 1.2、1.8、3.0、3.6。自取距离截断保持 800 m。",
        }
        doc.para(definitions[factor])
        doc.table("经济与求解", ["档位", "总成本\n元/h", "全部骑手\n元/h", "节省率\n%", "每单成本\n元/单", "Gap\n%", "耗时\nmin"], [
            [tier(r), num(r["total_cost_per_hour"]), num(r["all_courier_cost_per_hour"]), num(r["saving_pct"]), num(r["cost_per_order"], 3), num(r["gap_pct"], 3), num(r["solver_seconds"] / 60)] for r in subset
        ])
        doc.table("服务与设施", ["档位", "无人机量\n单/h", "份额\n%", "自取\n单/h", "到家\n单/h", "机队\n架", "开放\nL/H/E", "柜单元"], [
            [tier(r), num(r["drone_served_per_hour"]), num(r["drone_served_share_pct"]), num(r["pickup_per_hour"]), num(r["home_per_hour"]), num(r["fleet"]), opened(r), str(round(r["locker_units"]))] for r in subset
        ])
        time_rows, op_rows = [], []
        for r in subset:
            t = details[r["case"]]["times"]
            rho = [b["rho"] for b in bases[r["case"]] if b["open"]]
            time_rows.append([tier(r), num(t["mean_delivery"]), num(t["direct_same_served_OD"]), num(t["first_rider"]), num(t["pre_pickup_flight"]), num(t["loaded_flight"]), num(t["return_flight"]), num(sum(t.get(k, 0) for k in ("pre_pickup_swap", "loaded_swap", "return_swap")))])
            op_rows.append([tier(r), num(t["sync_wait"]), num(t["terminal"]), num(r["cross_base_service_pct"]), num(r["swap_visits_per_hour"]), num(r["service_using_swap_pct"]), num(r["locker_capacity_utilization_pct"]), f"{min(rho):.3f}–{max(rho):.3f}"])
        doc.table("配送与飞行时间（min）", ["档位", "配送\n完成", "同 OD\n纯骑手", "首段\n骑手", "取餐前\n飞行", "带载\n飞行", "返航\n飞行", "任务内\n换电"], time_rows)
        doc.table("同步、末端与运营", ["档位", "同步等餐\nmin", "末端\nmin", "跨基地\n服务 %", "换电\n次/h", "换电订单\n%", "柜利用率\n%", "开放基地\nρ 范围"], op_rows)
        doc.para("ρ 范围列示开放基地的最小值和最大值。跨基地服务指起点基地与返回基地不同的订单；换电订单比例按至少换电一次的服务订单计。任务内换电时间包含返航段，柜利用率按取餐与投递处理量除以已安装处理能力计算。")

    for name, title, caption in [
        ("performance_summary", "新增档位的成本、服务与机队", "四组新增试验的成本节省率、无人机服务份额与机队规模。虚线给出原基准的参考水平；空心标记表示 gap 超过 0.5%。图中仅新增档位有试验点。"),
        ("cost_fleet_composition", "新增档位的成本与机队构成", "每单成本以全部潜在订单率计算；机队按空闲、飞行、换电、操作及同步等待等占用分项表示。"),
        ("delivery_times", "新增档位的配送时间", "取餐后的配送完成时间与纯骑手时间。每个档位均按其无人机服务 OD 和流量作共同加权；首末端口径见第 1 节。"),
    ]:
        doc.page(title)
        doc.figure(f"figures/{name}.png", caption, height="0.80")

    for cap in (3, 7):
        doc.page(f"运行地图：每个 H 最多 {cap} 个柜单元")
        doc.figure(f"maps/r3_max_cabinets_{cap}/network.png", f"每个 H 的柜单元上限为 {cap} 时的运行网络。候选数量 H20/L4/E8；不叠加 OD 连线。", height="0.82")

    doc.md.extend([
        "\n## 增补文件\n", "[轮次三全指标 Excel](results_full.xlsx)",
        "[轮次三全指标 CSV](full_metrics.csv)", "[轮次三全部地图](maps_index.md)",
        "[新增结果 LaTeX 三线表](sensitivity_tables.tex)",
    ])
    preamble = r"""\documentclass[a4paper,zihao=5,fontset=fandol]{ctexart}
\usepackage[margin=18mm,headheight=15pt]{geometry}
\usepackage{booktabs,graphicx,caption,xcolor,fancyhdr,hyperref,amsmath}
\definecolor{ink}{HTML}{163A50}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=ink,pdftitle={灵敏度试验结果增补：轮次三}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 灵敏度试验结果增补}
\fancyhead[R]{\small 轮次三}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\ctexset{section={format=\Large\bfseries\color{ink},beforeskip=6pt,afterskip=10pt},subsection={format=\normalsize\bfseries,beforeskip=9pt,afterskip=5pt}}
\captionsetup{font=small,labelfont=bf,skip=5pt}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.12}
\begin{document}
{\LARGE\bfseries\color{ink} 灵敏度试验结果增补}\par
{\large 轮次三 · 南山区域 2}\par
{\small 汇总日期：2026 年 9 月 22 日}\vspace{3mm}
"""
    (out / "report_zh.tex").write_text(preamble + "\n".join(doc.tex) + "\n\\end{document}\n")
    (out / "report_zh.md").write_text("# 灵敏度试验结果增补：轮次三\n\n汇总日期：2026 年 9 月 22 日。\n" + "\n".join(doc.md))
    make_tables(out, rows)
    return out / "report_zh.tex"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(build(args.directory))
