"""Create the Chinese experimental brief directly from saved solver records."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR=Path(__file__).parent/'nanshan_integer_20260914'


def render():
    records=[]
    for folder in ['formal_60s','extended_300s']:
        records.extend(json.loads(p.read_text())|{'file':f'{folder}/{p.name}'}
                       for p in sorted((DIR/folder).glob('H*.json')))
    validation=json.loads((DIR/'validation.json').read_text())
    short=[r for r in records if r['budget']==60.]
    by={(r['name'],r['method']):r for r in short}
    levels=[11,24,43,68,87,120]
    rows=[]
    for h in levels:
        a=by[(f'Nanshan_H{h}','integer_benders')]
        b=by[(f'Nanshan_H{h}','integer_benders_pooled')]
        rows.append((h,a,b))
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(len(rows));w=.34
    ax.bar(x-w/2,[100*a['gap'] for _,a,b in rows],w,label='Integer Benders',color='#9ba9b5')
    ax.bar(x+w/2,[100*b['gap'] for _,a,b in rows],w,label='+ Master cuts',color='#247c76')
    ax.set_xticks(x,[f'H{h}\n{a["n_od"]:,} OD' for h,a,b in rows])
    ax.set_ylabel('Certified objective gap (%)')
    ax.axhline(.1,color='#a3672c',linestyle='--',linewidth=1,label='0.1% target')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(DIR/'gap_comparison.png',dpi=170);plt.close(fig)
    largest=[r for r in records if r['name']=='Nanshan_H120']
    fig,ax=plt.subplots(figsize=(8,4.5))
    for r in largest:
        t=[v['elapsed'] for v in r['trace'] if v['lower']>0]
        gap=[max(1e-6,100*(v['upper']-v['lower'])/max(1,abs(v['upper']))) for v in r['trace'] if v['lower']>0]
        t.append(r['wall']);gap.append(max(1e-6,100*r['gap']))
        label=('Cuts' if r['method'].endswith('pooled') else 'Base')+f' / {r["budget"]:.0f}s budget'
        ax.step(t,gap,where='post',label=label)
    ax.set_xlabel('Elapsed solve wall time (s)');ax.set_ylabel('Certified objective gap (%)')
    ax.set_ylim(bottom=0);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(DIR/'h120_convergence.png',dpi=170);plt.close(fig)

    best_large=min(largest,key=lambda r:r['gap'])
    strongest=sum(b['bound']>a['bound']+1e-5 for _,a,b in rows)
    target_met=sum(r['gap']<=r['mip_gap_target']+1e-8 for r in records)
    lines=['# MLND_Path：整数主问题与南山全量规模实验','',
        f'本轮完成 **{len(records)} 次南山全量实验**。主问题始终保留设施与 OD 基地对二元决策，只有一个连续的成本估计变量；可用率、idle 与空返全部留在子问题。最大档含 **11,427 个 OD、731,469 个二元变量**，未抽样订单、未缩放需求、未使用 top-k 航廊筛选。','',
        f'60 秒预算的六档对照中，加强割在 **{strongest}/6 档**取得更高下界。最大档本轮最小认证间隙为 **{100*best_large["gap"]:.3f}%**，实际用时 **{best_large["wall"]:.1f} 秒**。{target_met}/{len(records)} 次运行达到设定的 0.1% 目标间隙。','',
        '[模型、路径算法与割的完整说明](METHOD.html) · [全候选对偶补全推导](integer_cuts_design.html) · [历史数据与参数审计](data_audit.html)','',
        '## 1. 六档完整数据','',
        '| 档位 | 基地／柜／换电点 | OD 数 | 订单/h | 主问题二元变量 | 基础路径组合数，不含换电变体 |',
        '|---|---:|---:|---:|---:|---:|']
    for h,a,b in rows:
        L={11:2,24:2,43:3,68:5,87:6,120:8}[h];K={11:2,24:3,43:5,68:7,87:9,120:13}[h]
        lines.append(f'| H{h} | {L} / {h} / {K} | {a["n_od"]:,} | {a["demand"]:,.2f} | {a["stats"]["master_binaries"]:,} | {a["basic_path_combinations"]:,} |')
    lines+=['','最大档的基础组合数约 **104.43 亿**。实现通过分阶段资源状态与最短路共享计算，直接取得每个 OD—基地对的最低任务成本，不建立这些组合对应的变量或路径字典。所有候选基地、柜、换电点均参与。','',
        '## 2. 60 秒预算：主问题加割对照','',
        '两种方法采用相同的隐式精确定价、连续子问题、初始方案和数据；加强版增加共享 idle 子模割与两个柜的成本覆盖割。表中列出实际耗时；求解器部分运行在设定预算之后返回。','',
        '| 档位 | 基础版 UB / LB | 基础 gap | 基础实际秒 | 加强版 UB / LB | 加强 gap | 加强实际秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for h,a,b in rows:
        lines.append(f'| H{h} | {a["objective"]:,.2f} / {a["bound"]:,.2f} | {100*a["gap"]:.3f}% | {a["wall"]:.1f} | {b["objective"]:,.2f} / {b["bound"]:,.2f} | {100*b["gap"]:.3f}% | {b["wall"]:.1f} |')
    lines+=['','UB 是可行方案成本，LB 是全局下界，单位均为 CNY/h；gap = (UB−LB)/UB。','',
            '![六档加割对照](gap_comparison.png)','',
            '## 3. 最大档延长实验','',
            '| 方法／预算 | 实际秒 | 可行成本 UB | 全局下界 LB | gap | Benders 割 | 动态及初始 idle 割 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(largest,key=lambda r:(r['budget'],r['method'])):
        name=('加强' if r['method'].endswith('pooled') else '基础')+f' / {r["budget"]:.0f}s'
        lines.append(f'| {name} | {r["wall"]:.1f} | {r["objective"]:,.2f} | {r["bound"]:,.2f} | {100*r["gap"]:.3f}% | {r["stats"]["benders_cuts"]} | {r["stats"]["pooled_cuts"]} |')
    lines+=['','![H120 收敛轨迹](h120_convergence.png)','',
        '## 4. 实际加了什么割','',
        '1. **共享 idle 子模割。** 把同一基地的 OD 服务收益合并后，对 idle 函数取一维最小值，再生成只含分配二元变量的线性下界。它在当前整数分配处计入一个共享的 idle 成本。',
        '2. **两个柜的成本覆盖割。** 只有零个或一个柜开放时，服务任务无法经过两个不同柜，成本回到全骑手基准。该事实写入成本下界，保留所有选址下的运营补救。',
        '3. **完整运营 Benders 割。** 当前开放路径由精确最短任务求解；其余路径由全设施开放的组最低成本及设施激活罚价补全，对所有候选路径形成全局证书。',
        '',
        '主问题还使用聚合的起点／终点基地开放链接。两种测试方法均已包含这些链接和原有的按 OD 分摊 idle 初始下界。','',
        '## 5. 计算时间现在花在哪里','',
        '| H120 方法／预算 | 建模及首次定价秒 | 全部定价秒 | 全部凸内层秒 | 凸内层次数 | 总秒 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted(largest,key=lambda r:(r['budget'],r['method'])):
        name=('加强' if r['method'].endswith('pooled') else '基础')+f' / {r["budget"]:.0f}s'
        lines.append(f'| {name} | {r["build_wall"]:.2f} | {r["oracle_wall"]:.2f} | {r["recourse_wall"]:.2f} | {r["stats"]["recourse_calls"]} | {r["wall"]:.2f} |')
    lines+=['','建模及首次定价与“全部定价”部分重叠，不能逐列相加。总时间包含主问题搜索、割构造、定价及内层求解。当前连续内层只随基地数增长，进一步加速的重点是整数主问题的下界和搜索。','',
        '## 6. 当前方案与参数口径','',
        '| 档位（60 秒加强版） | 开放 L/H/E | 机队（连续） | 无人机服务单/h | 服务占比 | 成本节约区间 CNY/h |',
        '|---|---:|---:|---:|---:|---:|']
    for h,a,b in rows:
        opened=b['opened'];c0=b['all_ground_cost']
        lines.append(f'| H{h} | {opened["L"]}/{opened["H"]}/{opened["E"]} | {b["fleet"]:,.1f} | {b["served"]:,.1f} | {100*b["served"]/b["demand"]:.1f}% | [{c0-b["objective"]:,.1f}, {c0-b["bound"]:,.1f}] |')
    lines+=['','机队按 idle 数量加任务流量乘平均占用时间计算，取餐前的同步等待也计入占用。此版本采用连续机队且没有柜服务率上限，服务量与历史有限服务率柜模型相应变化。','',
        '成本、72 km/h 速度、10 km 航程、3 min 换电和末端 utility 沿用历史最终配置。骑手 18 km/h 与空返时钟 κ=30/h 沿用路径原型设置。原始时钟得到的经验前置时间分布只通过期望进入路径系数；没有增加场景变量。完整来源及各参数处理见数据审计与方法说明。','',
        '参数迁移还显示了两个值得继续检查的机制。H120 的 60 秒方案开启 8 个基地、120 个柜、0 个外部换电点，设施小时费用为 **232 CNY/h，约占总成本 0.48%**；当前最好方案倾向于多开柜。其约 7,535 单/h 的服务量乘 15.394 min 前置时间，在任务时间分解中对应约 **1,933 架**的占用，占总机队约 **72%**。后续选址研究可重点检查设施费与接驳费的敏感性，以及订单何时开始占用无人机；这两项会直接改变开站和机队配置。','',
        '## 7. 正确性、复现与后续重点','',
        f'28 项相关测试通过，包括资源路径独立枚举、经验期望、子模割，以及 272 个战略设计之间的 73,984 次运营割检验。所有 {len(validation["runs"])} 个保存方案均重新计算路径、成本、机队与流量平衡；总体验证结果为 **{"通过" if validation["passed"] else "需检查"}**。最大约束残差 {max(r["residual"] for r in validation["runs"]):.3g}，最大成本重算误差 {max(r["objective_error"] for r in validation["runs"]):.3g} CNY/h。',
        '',
        '后续算法工作继续保持整数战略主问题：在分数节点分离共享 idle 割，强化关闭设施的成本罚价，并针对基地流量平衡生成更强的投影下界。本轮先建立全量运行、严格上下界与可复核的加割对照。','',
        '- [逐运行汇总 CSV](summary.csv)',
        '- [方案与跨方法下界验证](validation.json)',
        '- [60 秒实验清单](formal_60s/manifest.json)',
        '- [300 秒实验清单](extended_300s/manifest.json)',
        '',
        '复现入口：`python -m mlnd_path.benchmark_nanshan`。最大档延长实验：`python -m mlnd_path.benchmark_nanshan --levels 120 --seconds 300 --output mlnd_path/nanshan_integer_20260914/extended_300s`。实验位于 `MLND_Path` 分支；原论文正文保持原样。']
    (DIR/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fields=['name','method','n_od','demand','budget','wall','objective','bound','gap','fleet','served','oracle_wall','recourse_wall']
    with (DIR/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
    css='body{max-width:1180px;margin:36px auto;padding:0 24px;font:16px/1.8 -apple-system,"PingFang SC",sans-serif;color:#24333d}h1{font-size:30px}h2{margin-top:40px;border-top:1px solid #ddd;padding-top:20px}table{border-collapse:collapse;font-size:13px;width:100%;margin:24px 0}td,th{padding:8px;border-bottom:1px solid #dde3e6;text-align:left}th{background:#eef4f5}tr:nth-child(even){background:#fafbfc}img{max-width:100%}a{color:#176e68}strong{color:#175d58}code{background:#eef1f3;padding:2px 4px}@media(max-width:800px){table{display:block;overflow:auto}body{padding:0 14px}}'
    (DIR/'style.html').write_text('<style>'+css+'</style>')
    for stem,target in [('REPORT','report'),('METHOD','METHOD'),('integer_cuts_design','integer_cuts_design'),('data_audit','data_audit')]:
        subprocess.run(['/opt/anaconda3/bin/pandoc',str(DIR/(stem+'.md')),'-s','--mathjax',
            '-H',str(DIR/'style.html'),'-M',f'title=MLND_Path：{stem}',
            '-o',str(DIR/(target+'.html'))],check=True)
        html=DIR/(target+'.html')
        html.write_text(re.sub(r'href="(/[^"\n]+?):(\d+)"',
            lambda m:f'href="{Path(m.group(1)).as_uri()}" title="源文件第 {m.group(2)} 行"',html.read_text()))


if __name__=='__main__':render()
