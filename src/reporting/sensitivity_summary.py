"""Reconcile the two saved sensitivity rounds and export their comparison data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.experiments.storage import ROOT, sha256, write_json


ROUND_PATHS = (
    ROOT / "outputs/experiments/h20_sensitivity_20260920/batch",
    ROOT / "outputs/experiments/sensitivity_round2_20260920/batch",
)
DEFAULT_OUTPUT = ROOT / "outputs/reports/sensitivity_rounds_1_2_20260920"
FACTORS = dict(demand="订单量", rider_cost="骑手成本", fleet_cost="无人机固定成本",
               speed="飞行速度", battery="电池续航", density="候选站点密度")


def collect():
    rows, sources = [], []
    for round_number, root in enumerate(ROUND_PATHS, 1):
        state = json.loads((root / "status.json").read_text())
        if state["status"] != "finished":
            raise ValueError(f"Round {round_number} is not finished")
        for job in state["jobs"]:
            path = Path(job["output"])
            report = json.loads((path / "sensitivity/sensitivity_metrics.json").read_text())
            run = json.loads((path / "summary.json").read_text())
            result = json.loads((path / "result.json").read_text())
            if not (job["state"] == "finished" and run["passed"]
                    and run["postprocess_status"] == "complete"):
                raise ValueError(f"Incomplete or unvalidated result: {path}")
            row = dict(report["summary"])
            row.update(round=round_number, run_path=str(path), job_id=job["id"],
                       solver_objective=result["objective"],
                       solver_termination_reason=result["termination_reason"],
                       termination_reason=job.get("stop_reason") or result["termination_reason"],
                       solver_gap=result["gap"], wall_minutes=result["wall"] / 60,
                       above_gap_target=result["gap"] > .005 + 1e-8)
            row["saving_upper_bound_pct"] = 100 * (
                row["all_courier_cost_per_hour"] - result["bound"]
            ) / row["all_courier_cost_per_hour"]
            row["cost_per_order"] = row["total_cost_per_hour"] / row["demand_per_hour"]
            row["fleet_per_100_drone_orders"] = (
                100 * row["fleet"] / row["drone_served_per_hour"]
                if row["drone_served_per_hour"] else None)
            assert abs(row["total_cost_per_hour"] - result["objective"]) < .001
            assert abs(row["saving_pct"] - 100 * (1 - row["total_cost_per_hour"]
                                                  / row["all_courier_cost_per_hour"])) < 1e-8
            assert abs(row["drone_served_share_pct"] - 100 * row["drone_served_per_hour"]
                       / row["demand_per_hour"]) < 1e-8
            rows.append(row)
            sources.append(dict(job_id=job["id"], round=round_number, files={
                str(path / name): sha256(path / name) for name in (
                    "input_arrays.npz", "input_metadata.json", "result.json",
                    "summary.json", "sensitivity/sensitivity_metrics.json")
            }))
    assert len(rows) == 25 and len({row["job_id"] for row in rows}) == 25
    return rows, sources


def grouped(rows):
    baseline = next(row for row in rows if row["factor"] == "baseline")
    result = []
    for factor in FACTORS:
        group = [dict(row, group=factor, shared_baseline=False)
                 for row in rows if row["factor"] == factor]
        group.append(dict(baseline, group=factor, shared_baseline=True))
        assert len(group) == 5
        for row in sorted(group, key=lambda row: row["multiplier"]):
            row["x"] = row["candidate_lockers"] if factor == "density" else row["multiplier"]
            result.append(row)
    return result


def _table(rows):
    lines = ["| 档位 | 总成本（元/h） | 节省率 % | 无人机份额 % | 机队 | 开放 L/H/E | Gap % |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for row in rows:
        label = (f"H{row['candidate_lockers']} L{row['candidate_bases']} E{row['candidate_swaps']}"
                 if row["group"] == "density" else f"{row['multiplier']:.2f} 倍")
        if row["shared_baseline"]:
            label += "（基准）"
        gap_mark = "†" if row["above_gap_target"] else ""
        lines.append(f"| {label} | {row['total_cost_per_hour']:,.2f} | {row['saving_pct']:.2f} | "
                     f"{row['drone_served_share_pct']:.2f} | {row['fleet']:.2f} | "
                     f"{row['opened_bases']}/{row['opened_lockers']}/{row['opened_swaps']} | "
                     f"{row['gap_pct']:.3f}{gap_mark} |")
    return "\n".join(lines)


def write_report(output, rows, plot_rows):
    baseline = next(row for row in rows if row["factor"] == "baseline")
    text = ["# 灵敏度分析：轮次一与轮次二\n",
            "两轮共 25 组算例，全部完成并通过运营校验。18 组达到 0.5% gap，"
            "7 组按一小时预算停止，最终 gap 为 0.56%–1.72%。六类分析共用一次基准求解。\n",
            "## 基准与口径\n",
            "南山区域 2，区域面积 66.67 km²；400 个 OD，需求 3,583.57 单/小时。"
            "候选站点为 4 个基地、20 个柜位置和 8 个换电站。飞行速度 72 km/h，"
            "电池续航 8.33 min；无人机固定成本 7.20 元/架·小时。固定投入按五年、"
            "每年 365 天、每天 12 小时折旧。每柜单元处理能力 30.4 次/小时，"
            "一个位置最多安装 5 个单元。\n",
            "基准总成本 **20,561.59 元/小时**；对应全部骑手配送成本 21,660.55 元/小时，"
            "节省 **1,098.97 元/小时（5.07%）**。无人机服务 **29.58%** 的订单，"
            "连续机队规模 **99.73**；开放 **4L/18H/5E**，安装 70 个柜单元。\n",
            "- 节省率的分母是该算例自身的全部骑手配送成本，随订单量、骑手费率更新。\n"
            "- `L/H/E` 在结果中依次表示开放的基地、柜位置和换电站数量；柜单元数另行保存。\n"
            "- 无人机固定成本仅缩放机队单价；骑手成本同时缩放纯骑手、首段和末段费用。\n"
            "- 速度实验固定电池续航时间，因此同时改变飞行时间与可达距离；电量实验缩放电池续航时间。\n"
            "- 密度实验固定区域、全部 400 个 OD 及需求，候选坐标逐档嵌套；实际开放数量由模型决定。\n"
            "- † 表示达到时间预算后保存的解；耗时记录包含终止与保存收尾。\n",
            "## 主要结果\n",
            "1. 骑手成本从 0.5 倍升至 1.5 倍，节省率由 **0.01% 升至 9.79%**，"
            "无人机份额由 **3.39% 升至 37.45%**。对应开放设施从 1L/4H/0E 增至 4L/20H/6E。\n"
            "2. 订单量从 0.5 倍升至 1.5 倍，无人机服务量由 **657.92 增至 1,282.38 单/小时**，"
            "但份额从 **36.72% 降至 23.86%**。柜单元由 44 增至 85；该组柜处理能力利用率"
            "均为 98%–100%，新增需求主要由纯骑手承担。\n"
            "3. 速度从 0.5 倍升至 1.5 倍，节省率从 **1.25% 升至 7.11%**；同范围续航变化"
            "对应 **3.99%–5.17%**。当前设置下，速度变化的收益幅度更大。\n"
            "4. 无人机固定成本从 1.5 倍降至 0.5 倍，机队由 **80.08 增至 151.07**，"
            "无人机份额由 **26.10% 增至 36.10%**，节省率由 **3.51% 增至 7.14%**。\n"
            "5. 候选 H 从 10 增至 20，节省率从 **0.40% 升至 5.07%**；"
            "H20、H25、H30 当前解均约 5%。三者成本界分别为 "
            "[20,458.84, 20,561.59]、[20,413.79, 20,567.26]、[20,205.87, 20,559.43] 元/小时。"
            "H25 当前成本比 H20 高 5.68 元/小时，来自限时求解的解差异。\n",
            "## 全部结果\n"]
    for factor, label in FACTORS.items():
        text += [f"### {label}\n", _table([r for r in plot_rows if r["group"] == factor]) + "\n"]
    text += ["## 重调度与换电\n",
             "基准空载重调度为 0；**23.89%** 的无人机订单跨基地运行。换电服务为 "
             "**326.14 次/小时**，**30.47%** 的无人机订单使用换电。\n",
             "空载调度超过 0.0001 次/小时的算例只有三个：速度 1.25 倍为 **0.1771**，"
             "续航 0.5 倍为 **0.6950**，H30 密度为 **4.5393** 次/小时。"
             "其余仅有零值或数值残差。\n",
             "换电总次数随服务量和路线选择共同变化。续航 0.5 倍时无人机服务 831.31 单/小时，"
             "换电 176.56 次/小时；基准服务 1,059.97 单/小时，换电 326.14 次/小时。"
             "完整 CSV 同时保留换电次数与使用换电的订单比例。\n",
             "## 图表\n",
             f"![六类因素的成本节省率]({output / 'figures/savings_sensitivity.png'})\n",
             f"![速度和续航对运营的影响]({output / 'figures/technology_sensitivity.png'})\n",
             f"![站点密度对网络的影响]({output / 'figures/density_sensitivity.png'})\n",
             "所有图均有同名 PDF 矢量版本。空心点表示最终 gap 超过 0.5%；"
             "节省率图的竖线给出由目标上下界换算的区间。\n",
             "## 交付文件\n",
             f"- [全部 25 组结果 CSV]({output / 'all_results.csv'})\n"
             f"- [LaTeX 三线表源码]({output / 'sensitivity_tables.tex'})\n"
             f"- [表格 PDF 预览]({output / 'tables_preview.pdf'})\n"
             f"- [全部算例地图索引]({output / 'maps.md'})\n"
             f"- [来源与指标定义]({output / 'source_manifest.json'})\n",
             "表格参考 Chen et al. (2025) 的 Table 5–6，以指标为行、参数档位为列；"
             "本表直接给出指标水平，成本采用每单成本，方便跨需求规模比较。\n"]
    (output / "report_zh.md").write_text("\n".join(text))
    maps = ["# 各算例的设施图与航线图\n", "地图均使用深圳实际底图，不叠加 OD 连线。\n"]
    for row in rows:
        folder = Path(row["run_path"]) / "sensitivity"
        maps.append(f"- 轮次{row['round']} · {row['scenario']}："
                    f"[设施图]({folder / 'facilities.png'}) / [航线图]({folder / 'network.png'})")
    (output / "maps.md").write_text("\n".join(maps) + "\n")
    assert baseline["candidate_lockers"] == 20


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = collect()
    plot_rows = grouped(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "all_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "all_results.json", rows)
    write_json(output / "plot_data.json", plot_rows)
    write_json(output / "source_manifest.json", dict(
        unique_cases=len(rows), grouped_plot_rows=len(plot_rows),
        at_gap_target=sum(not row["above_gap_target"] for row in rows),
        time_limited=sum(row["above_gap_target"] for row in rows),
        definitions=dict(
            saving_pct="100 * (same-scenario all-courier cost - total cost) / same-scenario all-courier cost",
            saving_upper_bound_pct="100 * (same-scenario all-courier cost - solver lower bound) / same-scenario all-courier cost",
            fleet="Continuous fleet-size decision Gamma",
            opened_lockers="Opened locker locations, not installed cabinet units",
            baseline="One saved baseline run shared across six factor groups",
            termination_reason="Batch stop reason if present; otherwise the solver termination reason"),
        sources=sources))
    write_report(output, rows, plot_rows)
    return dict(output=str(output), validated_cases=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    print(json.dumps(build(parser.parse_args().out), indent=2))
