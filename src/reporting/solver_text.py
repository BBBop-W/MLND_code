"""Solver settings and stage timings for current and archived experiment reports."""

from __future__ import annotations

from .formatting import _n, _tr

STAGE_NAMES = {
    "benders": ("Benders + column generation", "Benders与列生成"),
    "design_search": ("Integer design search", "整数设计搜索"),
    "refinement": ("Full-graph column-generation refinement", "完整图列生成复核"),
    "polish": ("Fixed-design polishing", "固定设计精修"),
    "direct": ("Complete-column reference model", "全列参考模型"),
}


def stage_name(name, lang="en"):
    return _tr(*STAGE_NAMES.get(name, (name, name)), lang)


def solver_section(solver, lang="en"):
    """Return a heading-free paragraph list and tables of recorded solver data."""
    from .formatting import _table

    translate = lambda en, cn: _tr(en, cn, lang)

    def budget(value):
        if value is None and solver.get("schema") in {"stages", "direct"}:
            return translate("Unlimited", "不设限")
        return _n(value, 1)

    rows = [
        (translate("Algorithm", "算法"), solver["method"]),
        (
            translate("Total solve budget, seconds", "求解总时间预算，秒"),
            budget(solver["budget"]),
        ),
        (
            translate("Elapsed solve time, seconds", "求解实际经过时长，秒"),
            _n(solver["wall"], 2),
        ),
    ]
    if solver.get("total_wall") is not None:
        rows.append(
            (
                translate(
                    "Whole-run time including preparation and validation, seconds",
                    "含预处理及核验的完整运行时长，秒",
                ),
                _n(solver["total_wall"], 2),
            )
        )
    if solver.get("cumulative_solver_seconds") is not None:
        rows.append(
            (
                translate(
                    "Sum of recorded stage times, seconds", "记录的各阶段时长合计，秒"
                ),
                _n(solver["cumulative_solver_seconds"], 2),
            )
        )
    rows += [
        (
            translate("Thread limit", "线程上限"),
            solver.get("threads") if solver.get("threads") is not None else "—",
        ),
        (
            translate("Random seed", "随机种子"),
            solver.get("seed") if solver.get("seed") is not None else "—",
        ),
        (
            translate("Relative MIP gap tolerance", "相对MIP Gap容差"),
            _n(solver.get("mip_gap"), 6),
        ),
        (
            translate("Absolute MIP gap tolerance", "绝对MIP Gap容差"),
            _n(solver.get("mip_gap_abs"), 6),
        ),
        (translate("Termination", "终止原因"), solver["termination_reason"] or "—"),
        (
            translate("Incumbent cost, CNY/h", "当前最好成本，元/小时"),
            _n(solver["objective"]),
        ),
        (
            translate("Full-model lower bound, CNY/h", "完整模型下界，元/小时"),
            _n(solver["bound"]),
        ),
        (
            translate("Full-model optimality gap", "完整模型Gap"),
            f"{_n(100 * solver['gap'] if solver['gap'] is not None else None)}%",
        ),
    ]
    for key, en, cn in [
        ("benders_cuts", "Benders cuts", "Benders割"),
        ("cg_iterations", "Column-generation iterations", "列生成迭代"),
    ]:
        if key in solver["stats"]:
            rows.append((translate(en, cn), solver["stats"][key]))
    parts = [_table([translate("Metric", "指标"), translate("Value", "数值")], rows)]
    stages = solver.get("stage_rows", [])
    if stages:
        parts.append(
            _table(
                [
                    translate("Stage", "阶段"),
                    translate("Budget (s)", "预算（秒）"),
                    translate("Actual wall time (s)", "实际时长（秒）"),
                    translate("Termination / status", "终止原因／状态"),
                ],
                [
                    (
                        stage_name(row["name"], lang),
                        budget(row.get("budget")),
                        _n(row.get("wall"), 2),
                        row.get("termination_reason") or row.get("status") or "—",
                    )
                    for row in stages
                ],
            )
        )
    if solver.get("schema") == "stages":
        parts.append(
            translate(
                "The recorded stages execute sequentially within the total solve budget. The elapsed solve clock includes coordination between stages; whole-run time additionally includes input preparation and independent validation.",
                "各阶段在求解总预算内依次执行。求解经过时长包含阶段间协调；完整运行时长另外包含输入准备和独立核验。",
            )
        )
    elif solver.get("timing_note"):
        parts.append(str(solver["timing_note"]))
    refinement = solver.get("full_cg_refinement") or {}
    if refinement:
        parts.append(
            _table(
                [translate("Refinement check", "复核指标"), translate("Value", "数值")],
                [
                    (
                        translate("Full-graph pricing", "完整图定价"),
                        refinement.get("full_graph_pricing", "—"),
                    ),
                    (
                        translate("Column generation complete", "列生成完成"),
                        refinement.get("cg_complete", "—"),
                    ),
                    (
                        translate("Refined incumbent accepted", "接受复核后的解"),
                        refinement.get("accepted", "—"),
                    ),
                ],
            )
        )
    search = solver.get("selection_heuristic") or {}
    statistics = search.get("stats") or search.get("statistics") or {}
    details = [
        (key, value)
        for key, value in statistics.items()
        if isinstance(value, (str, int, float, bool))
    ]
    if details:
        parts.append(
            _table(
                [
                    translate("Design-search statistic", "整数设计搜索统计"),
                    translate("Value", "数值"),
                ],
                details,
            )
        )
    return parts
