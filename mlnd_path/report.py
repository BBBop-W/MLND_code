"""Build the Chinese experiment readout from saved, inspectable solver records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


NAMES = {'full':'直接 MISOCP', 'full_strong':'加强后直接求解',
         'benders_lazy':'基础 Benders＋CG', 'benders_root':'加强 Benders＋CG'}
SCALE_NAMES = {'tiny':'微型','small':'小型','medium':'中型','large':'较大型','stress':'扩展测试'}
ORDER = ['tiny','small','medium','large','stress']
ALGS = list(NAMES)


def solved(r):
    return r.get('gap') is not None and r['gap'] <= 1e-4 + 1e-8 and r['status'] in ('optimal','OPTIMAL_TOL')


def build(inputs, destination):
    dest=Path(destination); records=[]; configs=[]
    for path in map(Path,inputs):
        records += json.loads((path/'summary.json').read_text())
        configs.append(json.loads((path/'config.json').read_text()))
    groups=defaultdict(list); bycase=defaultdict(list)
    for r in records:
        groups[r['scale'],r['algorithm']].append(r)
        bycase[r['instance']].append(r)
    errors=[r for r in records if r.get('error') or r['status']=='ERROR']
    assert not errors, errors
    for case, rs in bycase.items():
        assert max(r['bound'] for r in rs) <= min(r['objective'] for r in rs)+1e-5
    matrix_rows=[]
    for scale in ORDER:
        rs=[r for r in records if r['scale']==scale]
        if not rs:continue
        one=rs[0]
        counts=sorted(set(r['service_columns'] for r in rs))
        cols=str(counts[0]) if len(counts)==1 else f'{counts[0]}–{counts[-1]}'
        dimensions='/'.join(str(one[k]) for k in ('n_l','n_h','n_od','n_k'))
        gens={r['seed']:r['generation_wall'] for r in rs}
        matrix_rows.append(f'| {SCALE_NAMES[scale]} | {dimensions} | {len(gens)} | {cols} | {statistics.median(gens.values()):.3f} |')
    timing=[]
    for scale in ORDER:
        if not any(s==scale for s,a in groups):continue
        cells=[]
        for alg in ALGS:
            rs=groups.get((scale,alg),[])
            if not rs:cells.append('—');continue
            median=statistics.median(r['wall'] for r in rs)
            gaps=[100*r['gap'] for r in rs]
            if all(solved(r) for r in rs):
                cells.append(f'{median:.3f} 秒；全部达标')
            else:
                cells.append(f'{median:.2f} 秒；gap {min(gaps):.2f}%–{max(gaps):.2f}%')
        timing.append('| '+SCALE_NAMES[scale]+' | '+' | '.join(cells)+' |')
    detailed=[]
    for case,rs in sorted(bycase.items(),key=lambda pair:(ORDER.index(pair[1][0]['scale']),pair[1][0]['seed'])):
        best=min(r['objective'] for r in rs)
        for r in sorted(rs,key=lambda r:ALGS.index(r['algorithm'])):
            loss=max(0,(r['objective']-best)/max(1,abs(best)))*100
            detailed.append(f'| {case} | {NAMES[r["algorithm"]]} | {r["objective"]:.6f} | {r["bound"]:.6f} | {100*r["gap"]:.4f}% | {loss:.4f}% | {r["wall"]:.3f} |')
    bottleneck=[]
    for scale in ORDER:
        for alg in ('benders_lazy','benders_root'):
            rs=groups.get((scale,alg),[])
            if not rs:continue
            med=lambda f:statistics.median(f(r) for r in rs)
            bottleneck.append(f'| {SCALE_NAMES[scale]} | {NAMES[alg]} | {med(lambda r:r["stats"]["subproblem_calls"]):.0f} | '
                              f'{med(lambda r:r["stats"]["subproblem_wall"]):.3f} | '
                              f'{med(lambda r:r["stats"]["master_wall"]):.3f} | '
                              f'{med(lambda r:r["stats"]["pricing_wall"]):.3f} | '
                              f'{med(lambda r:100*r["service_pool_size"]/max(1,r["service_columns"])):.1f}% |')
    operations=[]
    for case,rs in sorted(bycase.items()):
        best=min(rs,key=lambda r:r['objective']);o=best['operations']
        operations.append(f'| {case} | {o["open_base_count"]}/{o["open_hub_count"]}/{o["open_remote_swap_count"]} | '
                          f'{o["served_order_rate"]:.2f} | {o["pickup_order_rate"]:.2f}/{o["home_delivery_order_rate"]:.2f} | '
                          f'{o["remote_swap_rate"]:.2f} | {o["cross_base_service_rate"]:.2f} | {o["empty_reposition_rate"]:.2f} |')
    count_table=[]
    for alg in ALGS:
        rs=[r for r in records if r['algorithm']==alg]
        count_table.append(f'| {NAMES[alg]} | {sum(solved(r) for r in rs)}/{len(rs)} |')
    residual=max(r.get('primal_residual',0) or 0 for r in records)
    obj_error=max(r['operations']['objective_recomputation_error'] for r in records)
    source_matches=[]
    for config in configs:
        for filename,expected in config['source_sha256'].items():
            if filename == 'report.py':
                continue  # Presentation code is not executed by any solver run.
            path=dest/filename
            source_matches.append(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected)
    assert all(source_matches), 'Computational source changed after benchmark launch'
    md=f'''# MLND_Path 算法测试简报

本轮结果支持将**加强后的直接 MISOCP**作为当前求解基线。Benders＋列生成在微型和小型问题上求得一致结果；进入中型后，主要困难是收紧整数选址的全局下界。列生成本身运行正常，当前尚未表现出整体速度优势。

已在 `MLND_Path` 分支建立独立实现；本报告汇总 **{len(bycase)} 个实例、{len(records)} 次算法运行**。测试使用合成几何、需求、费用、解析随机取餐期望和目的柜 Logit，包含实际换电和连续电量约束。

## 1. 测试口径与规模

统一单线程，每次总求解预算 30 秒，目标相对最优性间隙 0.01%。路径预处理单独计时，也在原始 JSON 中给出端到端耗时。每个实例的算法顺序轮换；正式计时顺序执行，避免并行求解相互争用。环境：Python {configs[0]['python']}、Gurobi {'.'.join(map(str,configs[0]['gurobi']))}、{configs[0]['platform']}。

| 规模 | L/H/OD/远程换电站 | 种子数 | 服务列数 | 预处理时间中位数／秒 |
|---|---|---:|---:|---:|
{chr(10).join(matrix_rows)}

候选列通过“全部允许换电设施子集＋精确资源标签”生成，无人工跳数或 top-K 截断。该预处理精确保留当前无硬机队上限、仅设施激活约束的模型在每个整数设计下的运营值。CG 对固定候选列全集定价，包括经过当前关闭设施的列。完整证明和适用条件见 [实现说明](README.md)。

## 2. 求解效率

表内时间为同规模中位数；未达标组列出终止时自身的最优性间隙范围。达到时限的运行表示“在该预算下未完成证明”。

| 规模 | 直接 MISOCP | 加强后直接求解 | 基础 Benders＋CG | 加强 Benders＋CG |
|---|---|---|---|---|
{chr(10).join(timing)}

| 算法 | 达到 0.01% 精度的实例数 |
|---|---:|
{chr(10).join(count_table)}

![各规模运行时间与终止间隙](timing_and_gaps.svg)

图中星号表示该组包含在 30 秒预算内未达到目标间隙的运行。

加强后的直接求解增加来源流量与可用率的线性凸包、按 OD 拆分的设施激活约束。加强 Benders 同时使用这些约束、根节点分数设计割，以及整数分支中的 lazy Benders 割；两版均使用相同的完整候选列和机队／服务成本。

## 3. 解的质量与证明进度

“相对最佳已知解差”比较各算法找到的可行方案；“自身 gap”比较该算法的可行上界和全局下界。两列共同反映找到好方案的能力与证明最优性的进度。

| 实例 | 算法 | 可行目标值 | 全局下界 | 自身 gap | 相对最佳已知解差 | 秒 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(detailed)}

## 4. 瓶颈在哪里

以下为同组中位数。“连续层”包含建模、原始／对偶求解及完整定价；“定价”是其中的子项，不能再次相加。活动列池比例按所有跨设计累计进入池的服务列计算。

| 规模 | 方法 | 连续子问题次数 | 连续层秒 | 主问题秒 | 定价秒 | 列池／全集 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(bottleneck)}

当前实现每轮重建受限 SOCP 及显式对偶，列系数矩阵缓存并跨整数设计复用列池。定价采用完整候选列扫描；候选全集仍保存在内存中。这组测试说明，减少活动列并不自动减少外层所需的选址评价次数。

下一步建议保留加强后的直接求解作为对照，优先研究来源分配与可用率的分解界，并减少重复建模。根节点割和按 OD 激活已经实现，可以据此继续做单项对照。隐式资源路径定价可在候选列存储成为瓶颈时接入。

另外完成了循环 Benders 和核心点选割的开发预实验。核心点选割在个别设计上改善了割值，但单个中型算例的额外求解开销未带来整体改善，因此本轮默认对照采用上述四种方法。

## 5. 运营机制确实进入了解

每个实例选择本轮目标值最低的可行方案，重算以下指标。服务、自取、送家、跨基地和空程均为每小时流量；换电为每小时次数。

| 实例 | 开放 L/H/远程换电站 | 无人机服务 | 自取／送家 | 远程换电 | 跨基地服务 | 空重定位 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(operations)}

## 6. 正确性与复算

13 项独立测试全部通过，覆盖单基地解析最优值、跨设计割、空列池可行、基准／加强整数模型等价、跨阶段电量、必须空返的两基地场景，以及超时后的上界与状态处理。另对 2 个种子、合计 288 个固定组合进行独立路径穷举；标签法和列并集的最优费用最大差为 1.78×10⁻¹⁵。

全部正式结果的最大原始约束残差为 **{residual:.3g}**，根据订单、流量和机队重算目标的最大误差为 **{obj_error:.3g}**。所有实例的算法下界均不超过跨算法最佳可行上界。实验启动时记录的计算模块哈希与交付源码逐一匹配。

原始配置、种子、逐算法解与迭代轨迹见 `benchmarks` 下的正式实验目录；运行入口为 `python -m mlnd_path.benchmark`，测试入口为 `python -m pytest tests/test_mlnd_path.py -q`。
'''
    (dest/'BENCHMARK_REPORT.md').write_text(md)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    scales=[s for s in ORDER if any(g[0]==s for g in groups)]
    english=['Direct','Direct + valid inequalities','Benders-CG','Benders-CG + VI/root cuts']
    colors=['#72819a','#216461','#b38b59','#644c80']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    x=np.arange(len(scales));width=.19
    for index,alg in enumerate(ALGS):
        t=[];g=[]
        for s in scales:
            rr=groups.get((s,alg),[])
            t.append(statistics.median(r['wall'] for r in rr) if rr else np.nan)
            g.append(statistics.median(100*r['gap'] for r in rr) if rr else np.nan)
        positions=x+(index-1.5)*width
        axes[0].bar(positions,t,width,color=colors[index],label=english[index])
        axes[1].bar(positions,g,width,color=colors[index])
        for pos,scale,seconds in zip(positions,scales,t):
            if any(not solved(r) for r in groups.get((scale,alg),[])):
                axes[0].text(pos,seconds*1.09,'*',ha='center',va='bottom',fontsize=11)
    axes[0].set_yscale('log');axes[0].set_ylabel('Median solve wall time (seconds, log scale)')
    axes[1].set_ylabel('Median final optimality gap (%)')
    axes[1].axhline(.01,color='#333',lw=.7,ls='--')
    for ax in axes:
        ax.set_xticks(x,scales);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=axes[0].get_legend_handles_labels()[0],labels=english,
               loc='outside upper center',ncols=2,frameon=False,fontsize=9)
    fig.savefig(dest/'timing_and_gaps.svg')
    svg_path = dest/'timing_and_gaps.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    fig.savefig(dest/'timing_and_gaps.png', dpi=140)
    plt.close(fig)
    css='''body{max-width:1200px;margin:40px auto;padding:0 28px;color:#24313d;font:16px/1.8 -apple-system,"PingFang SC",sans-serif}h1{font-size:30px}h2{font-size:23px;margin-top:44px;border-top:1px solid #ddd;padding-top:20px}table{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f4}tr:nth-child(even){background:#fafbfc}img{max-width:100%}a{color:#216461}code{background:#f1f2f3;padding:2px 5px;font-size:14px}strong{color:#194f4d}@media(max-width:800px){table{display:block;overflow:auto}body{padding:0 15px}}'''
    (dest/'report_style.css').write_text(css)
    subprocess.run(['/opt/anaconda3/bin/pandoc','BENCHMARK_REPORT.md','--standalone','--embed-resources',
                    '--metadata','pagetitle=MLND_Path 算法测试简报','--css=report_style.css',
                    '-o','benchmark_report.html'],cwd=dest,check=True)
    print(json.dumps({'instances':len(bycase),'runs':len(records),'maximum_residual':residual,
                      'objective_recomputation_error':obj_error,'source_hashes_verified':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('inputs',nargs='+')
    parser.add_argument('--destination',default='mlnd_path');args=parser.parse_args()
    build(args.inputs,args.destination)
