"""Frozen comparison of double-task support and joint-base cost cuts.

Only four new solves are scheduled. Previous assignment-support results remain
read-only and are revalidated before being copied into the reference snapshot.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil

from src.experiments import assignment_support_benchmark_20261022 as assignment
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json


ROOT = base.ROOT
MODULE = "src.experiments.branch_joint_benchmark_20261022"
DEFAULT_BATCH = ROOT / "outputs/branch_joint_benchmark_20261022"
DEFAULT_REFERENCE = ROOT / "outputs/assignment_support_benchmark_20261022"
VARIANTS = (("lifted_branch_joint", "benders_lifted"), ("conditional_branch_joint", "benders_conditional"))
NEW_SOLVER_ARGS = ("--double-support-cuts", "--joint-base-cost-cuts", "--branch-joint-seconds", "0.25",
                   "--branch-joint-preprocessing-seconds", "2", "--branch-joint-floor", "0.0001",
                   "--branch-joint-cut-limit", "2000", "--branch-joint-max-cuts", "16")
_ASSIGNMENT_COMMAND = assignment.command_for
_ASSIGNMENT_RECORD = assignment._record
_BASE_GUARD = base.guarded_execute
_BASE_REPORT = base.report


def jobs_for(origins):
    return [dict(id=f"{origin['id']}_K2_{variant}", variant=variant, instance=origin["instance"],
                 K=2, method=method, tight_airline_bounds=True, topology_cuts=True,
                 assignment_support_cuts=True, branch_joint_cuts=True,
                 output=f"runs/{origin['id']}/{variant}/K2")
            for origin in origins for variant, method in VARIANTS]


def command_for(batch, manifest, job, attempt):
    return _ASSIGNMENT_COMMAND(batch, manifest, job, attempt) + manifest["protocol"]["new_solver_args"]


def _reference_snapshot(reference):
    reference = Path(reference).resolve()
    manifest = base.verify(reference)
    if manifest.get("schema") != "assignment-support-benchmark/1":
        raise ValueError("This comparison requires the completed assignment-support benchmark")
    state = base.read(reference / "state.json")["jobs"]
    rows, traces = [], []
    for job in manifest["jobs"]:
        saved = state[job["id"]]
        if saved["status"] in ("pending", "running"):
            raise ValueError(f"Reference job is unfinished: {job['id']}")
        row = _ASSIGNMENT_RECORD(reference, job, saved)
        if not row["validation_passed"]:
            raise ValueError(f"Reference validation/hash failure: {job['id']}")
        row["reference_batch"] = str(reference)
        rows.append(row)
        progress = reference / saved["directory"] / "progress.jsonl"
        if not progress.is_file():
            raise ValueError(f"Reference trajectory missing: {job['id']}")
        traces.append((job["id"], progress))
    expected = {f"{origin['id']}_K2_{variant}" for origin in manifest["input_origins"]
                for variant, _ in assignment.VARIANTS}
    if {row["job"] for row in rows} != expected:
        raise ValueError("The reference does not contain the required four settings")
    return manifest, rows, traces


def prepare(batch, instances, *, reference=DEFAULT_REFERENCE, new_solver_args=None, binary=None):
    new_solver_args = NEW_SOLVER_ARGS if new_solver_args is None else new_solver_args
    if not isinstance(new_solver_args,(list,tuple)) or not new_solver_args or not all(isinstance(value,str) for value in new_solver_args):
        raise ValueError("Explicit reviewed new_solver_args are required before freezing")
    if any(value.startswith("--") and not value.startswith(("--branch-","--joint-","--double-")) for value in new_solver_args):
        raise ValueError("New cut options may not override the existing solver settings")
    reference = Path(reference).resolve()
    previous, rows, traces = _reference_snapshot(reference)
    cfg = previous["protocol"]
    required = dict(seconds_per_job=600, threads_per_job=1, parallel_jobs=4, seed=43,
                    relative_gap_target=.005, topology_seconds=2, topology_cut_limit=1000,
                    assignment_support_seconds=.25, assignment_support_cut_limit=2000,
                    assignment_support_max_cuts=16, assignment_support_floor=1e-4)
    if any(cfg.get(key) != value for key,value in required.items()):
        raise ValueError("Reference protocol differs from the agreed prior settings")
    old_inputs = {origin["id"]:origin for origin in previous["input_origins"]}
    for value in instances:
        path = Path(value).resolve()
        if old_inputs.get(path.name,{}).get("manifest_sha256") != base.sha256(path / "manifest.json"):
            raise ValueError(f"Input is not byte-identical to reference: {path}")
    limits = cfg["resource_limits"]
    base.prepare(batch, instances, binary=binary, seconds=600, threads=1, parallel=4, gap=.005, seed=43,
        job_rss_gib=limits["job_rss_bytes"]/1024**3, batch_rss_gib=limits["batch_rss_bytes"]/1024**3,
        system_available_gib=limits["system_available_floor_bytes"]/1024**3,
        hard_wall_seconds=cfg["hard_wall_seconds"], resource_sample_seconds=limits["sample_seconds"],
        terminate_grace_seconds=limits["terminate_grace_seconds"])
    batch = Path(batch).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["schema"] = "branch-joint-benchmark/1"
    manifest["jobs"] = jobs_for(manifest["input_origins"])
    manifest["protocol"] = dict(cfg, comparison="previous assignment support versus adding double-task and joint-base cuts",
        new_solver_args=list(new_solver_args), same_binary_as_reference=False, cross_batch_single_seed=True,
        branch_joint_cuts=True, full_reference=False, budget_includes_new_preprocessing=True)
    for origin in manifest["input_origins"]:
        prior = old_inputs[origin["id"]]
        for key in ("size", "all_courier_cost"):
            origin[key] = prior.get(key)
    target = batch / "reference"
    target.mkdir()
    for source, name in ((reference / "experiment_manifest.json","experiment_manifest.json"),
                         (reference / "analysis/comparison.json","previous_report.json"),
                         (reference / "reference/validated_comparison.json","earlier_reference.json")):
        shutil.copy2(source,target/name)
    write_json(target / "validated_comparison.json",dict(protocol=cfg,rows=rows))
    trace_meta = []
    for job_id, source in traces:
        copied = target / "progress" / job_id / "progress.jsonl"
        copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,copied)
        trace_meta.append(dict(job=job_id,source=str(source),copied=str(copied.relative_to(batch)),sha256=base.sha256(copied)))
    manifest["reference"] = dict(source=str(reference),revalidated_utc=base.existing.utc(),
        manifest_sha256=base.sha256(reference / "experiment_manifest.json"),progress=trace_meta,
        previous_report_sha256=base.sha256(reference / "analysis/comparison.json"))
    manifest["frozen_sha256"].update({str(path.relative_to(batch)):base.sha256(path)
                                    for path in sorted(target.rglob("*")) if path.is_file()})
    write_json(batch / "experiment_manifest.json",manifest)
    write_json(batch / "state.json",dict(status="prepared_not_started",updated_utc=base.existing.utc(),
        jobs={job["id"]:dict(status="pending",attempts=[]) for job in manifest["jobs"]}))
    (batch / "run.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\n"
        "main([*sys.argv[1:], '--batch', str(batch)])\n",encoding="utf-8")
    return dict(batch=str(batch),jobs=len(manifest["jobs"]),solver_started=False,
                reference_unchanged=True,input_generation=False)


def _guarded_execute(batch,manifest,job_id,command,attempt,cancel,**kwargs):
    routed = list(command)
    if "-m" in routed:
        index = routed.index("-m")+1
        if routed[index] in (base.__name__, assignment.MODULE, assignment.topology.MODULE):
            routed[index] = MODULE
    return _BASE_GUARD(batch,manifest,job_id,routed,attempt,cancel,**kwargs)


def _record(batch,job,state):
    row = _ASSIGNMENT_RECORD(batch,job,state)
    # The inherited reader expects the older strengthened reference; use the
    # direct previous assignment-support cohort for every reported comparison.
    for key in list(row):
        if key.startswith(("previous_strengthened_","change_from_previous_")):
            del row[key]
    source = Path(batch)/state.get("result_directory",state.get("directory","_missing"))
    result = base.optional_read(source / "result.json")
    row["branch_joint_cuts_requested"] = job["branch_joint_cuts"]
    for prefix in ("branch_joint", "branch_support", "joint_base", "double_support"):
        summary = result.get(prefix+"_summary") or {}
        for key,value in summary.items():
            row[prefix+"_"+key] = json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
        for suffix in ("preprocessing", "metadata"):
            if prefix+"_"+suffix in result:
                row[prefix+"_"+suffix+"_json"] = json.dumps(result[prefix+"_"+suffix],sort_keys=True)
                if suffix=="preprocessing" and isinstance(result[prefix+"_"+suffix],dict):
                    for key in ("support_complete","numerical_domain_safe","timed_out","completed_groups",
                                "total_groups","candidates_checked","cached_double_paths"):
                        row[prefix+"_preprocessing_"+key] = result[prefix+"_"+suffix].get(key)
        for suffix in ("cut_records", "attempts"):
            if isinstance(result.get(prefix+"_"+suffix),list):
                row[prefix+"_"+suffix+"_record_count"] = len(result[prefix+"_"+suffix])
    for key,value in (result.get("acceleration_statistics") or {}).items():
        if key.startswith(("branch_joint", "branch_support", "joint_base", "double_support")):
            row[key] = json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
    if isinstance(result.get("branch_joint_cut_records"),list):
        row["branch_joint_double_cut_records"] = sum(item.get("kind")=="double_branch" for item in result["branch_joint_cut_records"])
        row["branch_joint_joint_cut_records"] = sum(item.get("kind")=="joint_bases" for item in result["branch_joint_cut_records"])
    applied = row.get("continuous_root_applied_lower_bound")
    row["global_lb_minus_applied_root"] = assignment._difference(row["lower_bound"],applied)
    row["integer_lb_minus_applied_root"] = assignment._difference(row.get("integer_master_best_lower_bound"),applied)
    tol = 1e-7*max(1,abs(applied)) if assignment._number(applied) else None
    row["applied_root_comparison_tolerance"] = tol
    difference = row["global_lb_minus_applied_root"]
    row["global_lb_exceeds_applied_root"] = difference>tol if assignment._number(difference) else None
    reference = base.optional_read(Path(batch)/"reference/validated_comparison.json")
    previous = next((old for old in reference.get("rows",[]) if old["instance"]==job["instance"]
                    and old["method"]==job["method"]),{})
    row["reference_job"] = previous.get("job")
    for key in ("lower_bound","upper_bound","gap","continuous_root_lower_bound","continuous_root_applied_lower_bound",
                "integer_master_best_lower_bound","solve_seconds","integer_socp_calls","integer_master_calls",
                "topology_seconds","assignment_support_seconds","assignment_support_cuts",
                "assignment_support_skipped_integer_subproblems","all_courier_candidate_ratio"):
        row["previous_assignment_"+key] = previous.get(key)
        row["change_from_previous_"+key] = assignment._difference(row.get(key),previous.get(key))
    if row.get("all_paths") is None and previous:
        for key in ("all_paths","all_service_paths","all_empty_paths","initial_segments"):
            row[key] = previous.get(key)
        row["full_path_count_source"] = "reference/"+str(previous.get("full_path_count_source"))
    return row


@contextmanager
def adapters():
    replacements = dict(command_for=command_for,_record=_record,guarded_execute=_guarded_execute,report=_report)
    previous = {key:getattr(base,key) for key in replacements}
    try:
        for key,value in replacements.items():
            setattr(base,key,value)
        yield
    finally:
        for key,value in previous.items():
            setattr(base,key,value)


def _difference_display(value):
    return "—" if value is None else "0.000" if abs(value)<1e-8 else f"{value:,.6f}"


def _report(batch):
    outcome = _BASE_REPORT(batch)
    batch = Path(batch)
    data = base.read(batch / "analysis/comparison.json")
    previous = base.read(batch / "reference/validated_comparison.json")["rows"]
    rows,cfg = data["rows"],data["protocol"]
    complete = all(row["scheduler_status"] not in ("pending","running") for row in rows)
    fmt,table = assignment.topology._fmt,assignment.topology._table
    improved = sum(row.get("global_lb_exceeds_applied_root") is True for row in rows)
    lines = ["# 双单路径支持与跨基地联合割"+("" if complete else "（尚未完成）"),"",
        f"已终止 {sum(row['scheduler_status'] not in ('pending','running') for row in rows)}/{len(rows)}；"
        f"独立验证通过 {sum(bool(row['validation_passed']) for row in rows)}/{len(rows)}；"
        f"认证全局 LB 超过实际采用连续根界 {improved}/{len(rows)}。","",
        "本轮只新增双单路径支持与跨基地联合成本割，保留上一轮 assignment-support 的全部参数。"
        "L01/L02、K=2、模型、需求重构、运营限制和标准输入保持原样。"
        "每组 600 秒、四组并发、单线程、seed=43、gap=0.005；新预处理和分离均计入预算。"
        "这是跨批次单种子比较，资源竞争可能影响限时搜索，不据单次结果宣称普遍提速。","",
        "## 下界变化","",
        "以实际加入主问题的连续根界 applied 为主基准；原始锥界 raw 和原始差值另列。"
        "是否越过根界使用 1e-7 × max(1, |applied|) 的报告容差。",""]
    lines += table(["实例","方法","原 LB","新 LB","LB 增加","根界 applied","LB−根界","整数主问题最好 LB"],
        [(Path(row["instance"]).name,row["method"],row.get("previous_assignment_lower_bound"),row["lower_bound"],
          _difference_display(row.get("change_from_previous_lower_bound")),row.get("continuous_root_applied_lower_bound"),
          _difference_display(row.get("global_lb_minus_applied_root")),row.get("integer_master_best_lower_bound")) for row in rows])
    lines += ["","## 可行解、gap 与时间","",
        "每组 gap 只使用该组的 UB/LB；不拼接不同方法的最好界。缺失值显示 —，不视作零。",""]
    lines += table(["实例","方法","原 UB","新 UB","原 gap","新 gap","预算耗时 s","进程墙钟 s","终止","验证"],
        [(Path(row["instance"]).name,row["method"],row.get("previous_assignment_upper_bound"),row["upper_bound"],
          fmt(row.get("previous_assignment_gap"),percent=True),fmt(row["gap"],percent=True),row.get("solve_seconds"),
          row.get("external_wall_seconds"),row.get("termination_reason"),row["validation_passed"]) for row in rows])
    lines += ["","## 新割诊断","",
        "以下按原始诊断字段保留双单和联合割的数量及开销。数值门限只筛选可安全生成的割，不修改需求、概率或运营假设。",""]
    lines += table(["实例","方法","双单割","联合割","新割合计","尝试","证书","跳过 SOCP","数值筛选跳过"],
        [(Path(row["instance"]).name,row["method"],row.get("branch_joint_double_cuts"),row.get("branch_joint_joint_cuts"),
          row.get("branch_joint_cuts"),row.get("branch_joint_attempts"),row.get("branch_joint_certificates"),
          row.get("branch_joint_skipped_integer_subproblems"),row.get("branch_joint_numerical_skips")) for row in rows])
    lines += [""]
    lines += table(["实例","方法","预处理 s","新分离合计 s","完整支持域","初始完成组","最终完成组","全部组","初始缓存双单路径","双单割记录数","联合割记录数"],
        [(Path(row["instance"]).name,row["method"],row.get("branch_joint_preprocessing_seconds"),row.get("branch_joint_seconds"),
          row.get("branch_joint_preprocessing_support_complete"),
          row.get("branch_joint_preprocessing_completed_groups"),row.get("branch_joint_completed_double_groups"),
          row.get("branch_joint_preprocessing_total_groups"),
          row.get("branch_joint_preprocessing_cached_double_paths"),row.get("branch_joint_double_cut_records"),row.get("branch_joint_joint_cut_records"))
         for row in rows])
    lines += ["","新分离耗时为两种割共同检查的总耗时；未提供单独计时时不作人为拆分。"
        "初始预处理预算为 2 秒，尚未缓存的双单组可在后续分离中补全；初始完成组数不足不视为求解失败。"
        "数值筛选跳过为跨候选累计数，不等于不同 OD 的数量。"]
    lines += [""]
    lines += table(["实例","方法","整数 SOCP 次数","全骑手候选比例","原拓扑检查 s","新拓扑检查 s","主问题优化 s","子问题总 s"],
        [(Path(row["instance"]).name,row["method"],row.get("integer_socp_calls"),fmt(row.get("all_courier_candidate_ratio"),percent=True),
          row.get("previous_assignment_topology_seconds"),row.get("topology_seconds"),row.get("master_optimize_seconds"),
          row.get("subproblem_wall_seconds")) for row in rows])
    lines += ["","全骑手候选比例以实际求解的整数 SOCP 为分母，成本割提前拦截的候选不计入分母。","",
              "## 模型和路径规模","","规模依次为变量/线性约束/锥。完整路径数来自同输入的完整枚举参考，生成列数另列。",""]
    def size(row,name):
        return "/".join(fmt(row.get(name+"_"+key)) for key in ("variables","linear_rows","cones"))
    lines += table(["实例","方法","主问题","整数子问题峰值","完整服务/空飞路径","已生成服务/空飞列"],
        [(Path(row["instance"]).name,row["method"],size(row,"master"),size(row,"subproblem_peak"),
          f"{fmt(row.get('all_service_paths'))}/{fmt(row.get('all_empty_paths'))}",
          f"{fmt(row.get('generated_service_columns'))}/{fmt(row.get('generated_empty_columns'))}") for row in rows])
    lines += ["","## 原始根界与上轮参考",""]
    def precise(value):
        return "—" if value is None else f"{value:.9f}"
    def scientific(value):
        return "—" if value is None else f"{value:.6g}"
    lines += table(["任务","root raw","root applied","LB−raw","LB−applied","资源原因"],
        [(row["job"],precise(row.get("continuous_root_lower_bound")),precise(row.get("continuous_root_applied_lower_bound")),
          scientific(row.get("global_lb_minus_continuous_root")),scientific(row.get("global_lb_minus_applied_root")),
          row.get("resource_limit_kind")) for row in rows])
    lines += [""]
    lines += table(["实例","上轮方法","UB","LB","gap","预算耗时 s"],
        [(Path(row["instance"]).name,row["method"],row["upper_bound"],row["lower_bound"],fmt(row["gap"],percent=True),row.get("solve_seconds"))
         for row in previous])
    lines += ["","## 证据","",
        "实际新增参数：`"+" ".join(cfg["new_solver_args"])+"`。","",
        "[比较 CSV](comparison.csv) / [JSON](comparison.json)；"
        "[下界曲线 PDF](figures/lower_bound_comparison.pdf) / [PNG](figures/lower_bound_comparison.png)。"
        "曲线只在四组均终止且验证通过后生成。","",
        "代码与输入的冻结哈希见 `../experiment_manifest.json`，上轮只读快照见 `../reference/`。"
        "独立验证和资源记录见 `../runs/`。单组 RSS 4 GiB、批次合计 8 GiB、系统可用内存保底 2 GiB，"
        "每组含验证的外部墙钟上限 720 秒；资源终止不表示数学不可行。"]
    (batch / "analysis/REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return dict(outcome,complete=complete,bounds_beyond_applied_root=improved)


def plot(batch):
    from src.reporting.scale_ladder import _progress
    from src.reporting.style import _style,_grid,plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")["jobs"]
    rows = base.read(batch / "analysis/comparison.json")["rows"]
    if any(row["scheduler_status"] in ("pending","running") or not row["validation_passed"] for row in rows):
        raise ValueError("All four new runs must terminate and pass validation before plotting")
    previous = base.read(batch / "reference/validated_comparison.json")["rows"]
    selected = [(row,batch/state[row["job"]]["directory"],"new") for row in rows]
    selected += [(row,batch/"reference/progress"/row["job"],"previous") for row in previous]
    points,identities,rejected = [],[],[]
    for row,attempt,cohort in selected:
        tracking = dict(id=row["job"],level=Path(row["instance"]).name,K=2,method=row["method"],state="validated",upper_bound=row["upper_bound"])
        accepted,refused = _progress(dict(row=tracking,attempt=attempt))
        rejected.extend(refused)
        if not accepted:
            raise ValueError(f"No global-bound trajectory for {row['job']}")
        for point in accepted:
            point.update(cohort=cohort,point_source="program_log")
        final_time = max(float(row.get("solve_seconds") or 0),accepted[-1]["elapsed_seconds"])
        final = dict(**{key:tracking[key] for key in ("id","level","K","method")},state="validated",line=None,
            elapsed_seconds=final_time,upper_bound=row["upper_bound"],lower_bound=row["lower_bound"],gap=row["gap"],
            event="independently_validated_final_result",evidence="validated_comparison_row",cohort=cohort,point_source="validated_final")
        points += [*accepted,final]
        identities.append(dict(job=row["job"],cohort=cohort,progress_sha256=base.sha256(attempt/"progress.jsonl"),
            result_sha256=row["result_sha256"],final_lower=row["lower_bound"]))
    output = batch/"analysis/figures"
    output.mkdir(exist_ok=True)
    curve = batch/"analysis/global_bound_progress.csv"
    with curve.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    _style()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,axes = plt.subplots(1,2,figsize=(11.8,4.8))
    colors = {"benders_lifted":"#386b91","benders_conditional":"#a95779"}
    limits = {}
    for ax,level in zip(axes,("L01","L02")):
        roots = [row["continuous_root_applied_lower_bound"] for row in rows if Path(row["instance"]).name==level
                 and assignment._number(row.get("continuous_root_applied_lower_bound"))]
        values,finals = [],[]
        for method,color in colors.items():
            for cohort,style in (("previous","--"),("new","-")):
                subset = sorted((point for point in points if point["level"]==level and point["method"]==method and point["cohort"]==cohort),
                                key=lambda point:point["elapsed_seconds"])
                ax.step([point["elapsed_seconds"] for point in subset],[point["lower_bound"] for point in subset],
                        where="post",color=color,linestyle=style,linewidth=1.5)
                final = next(point for point in subset if point["point_source"]=="validated_final")
                ax.plot(final["elapsed_seconds"],final["lower_bound"],color=color,marker="o",markersize=4,
                        markerfacecolor=color if cohort=="new" else "white")
                finals.append(final["lower_bound"])
                values.extend(point["lower_bound"] for point in subset if point["elapsed_seconds"]>=30)
        values.extend(roots)
        if roots:
            ax.axhline(max(roots),color="#777777",linestyle=":",linewidth=.9)
            values = [value for value in values if value>=min(roots)-1e-7*max(1,abs(min(roots)))]
        values.extend(finals)
        lo,hi = min(values),max(values)
        pad = max((hi-lo)*.12,abs(hi)*.0001,.01)
        ax.set_ylim(lo-pad,hi+pad)
        ax.set_xlim(0,manifest["protocol"]["seconds_per_job"]*1.025)
        ax.set_title(f"{level}: expanded lower-bound scale",loc="left")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Certified global lower bound (CNY/h)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        _grid(ax,"both")
        limits[level] = dict(y_min=lo-pad,y_max=hi+pad,applied_roots=roots)
    handles = [Line2D([0],[0],color=color,label=label) for color,label in
               ((colors["benders_lifted"],"Lifted"),(colors["benders_conditional"],"Conditional"))]
    handles += [Line2D([0],[0],color="#333333",linestyle=style,label=label) for style,label in
                (("--","Previous assignment-support cuts"),("-","With double-task and joint-base cuts"),(":","Continuous root (applied)"))]
    fig.legend(handles=handles,loc="lower center",ncol=3,bbox_to_anchor=(.5,.045),frameon=False)
    fig.suptitle("Double-task and joint-base cuts: certified lower bounds",y=.98)
    fig.text(.5,.015,"Expanded cost scales; early points may fall below the axes. Final markers are independently validated.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.86,bottom=.27,wspace=.28)
    for extension in ("png","pdf"):
        fig.savefig(output/("lower_bound_comparison."+extension),dpi=240,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    write_json(batch/"analysis/figure_audit.json",dict(identities=identities,rejected_progress=rejected,accepted_points=len(points),
        final_values="independently validated rows",continuous_root_reference="applied_total_cost_lower_bound",axis_limits=limits,
        curve_csv_sha256=base.sha256(curve)))
    return dict(figure=str(output/"lower_bound_comparison.pdf"),accepted_points=len(points))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","verify","run","launch","report","status","plot","audit-worker"))
    parser.add_argument("--batch",type=Path,default=DEFAULT_BATCH)
    parser.add_argument("--reference",type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--new-solver-args-json",help="Reviewed JSON array appended to the previous native command")
    parser.add_argument("--jobs",nargs="+")
    parser.add_argument("--retry-failed",action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt",type=Path)
    args = parser.parse_args(argv)
    with adapters():
        if args.action=="prepare":
            result = prepare(args.batch,[ROOT/f"outputs/tsl_scale_ladder/instances/L{level:02d}" for level in (1,2)],
                reference=args.reference,binary=args.binary,
                new_solver_args=json.loads(args.new_solver_args_json) if args.new_solver_args_json else None)
        elif args.action=="verify":
            result = dict(passed=bool(base.verify(args.batch)),solver_started=False)
        elif args.action=="run":
            result = base.run(args.batch,only=args.jobs,retry_failed=args.retry_failed)
        elif args.action=="launch":
            result = base.launch(args.batch)
        elif args.action=="report":
            result = _report(args.batch)
        elif args.action=="plot":
            result = plot(args.batch)
        elif args.action=="audit-worker":
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(args.batch,args.job_id,args.attempt)
        else:
            result = base.existing.status_summary(args.batch)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
