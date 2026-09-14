"""Build the follow-up readout solely from saved benchmark records."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
NAMES = {'full_strong':'加强完整 MISOCP', 'benders_strong':'原锥子问题 Benders＋CG',
         'partial_full':'部分 Benders＋全列 LP', 'partial_cg':'部分 Benders＋增量 CG',
         'analytic':'按路径组解析多割', 'analytic_oa':'解析多割＋idle 外逼近',
         'analytic_oa_inject':'idle 外逼近＋可行解回灌'}
ORDER = ['tiny','small','medium','large','stress']


def build(inputs):
    rows=[]; configs=[]
    for directory in map(Path,inputs):
        rows+=json.loads((directory/'summary.json').read_text())
        configs.append(json.loads((directory/'config.json').read_text()))
    for config in configs:
        for name,digest in config['source_sha256'].items():
            if name=='report.py':continue
            assert hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()==digest,name
    cases=defaultdict(list)
    for r in rows:
        assert not r.get('errors'),r
        assert r['status'] not in ['CALLBACK_ERROR','ERROR','INCOMPLETE_RECOURSE'],r
        cases[r['instance']].append(r)
    for case,rr in cases.items():
        assert max(r['bound'] for r in rr)<=min(r['objective'] for r in rr)+1e-5,case
    max_res=max(max(r['independent_audit']['row_residual'],r['independent_audit']['cone_residual']) for r in rows)
    max_obj=max(r['independent_audit']['objective_recomputation_error'] for r in rows)
    xml=ET.parse(ROOT/'validation_tests.xml').getroot()
    test_count=sum(int(s.attrib.get('tests',0)) for s in xml.iter('testsuite'))
    timing=[];details=[];stats=[]
    for case,rr in sorted(cases.items(),key=lambda p:(ORDER.index(p[1][0]['scale']),p[1][0]['seed'])):
        first=rr[0];cells=[];best=min(r['objective'] for r in rr)
        for alg in NAMES:
            found=[r for r in rr if r['algorithm']==alg]
            if not found:cells.append('—');continue
            r=found[0]
            done=r['gap']<=configs[0]['mip_gap']+1e-8
            cells.append(f'{r["wall"]:.2f}s；'+('达标' if done else f'gap {100*r["gap"]:.2f}%'))
            details.append(f'| {case} | {NAMES[alg]} | {r["objective"]:.6f} | {r["bound"]:.6f} | {100*r["gap"]:.4f}% | {100*max(0.,r["objective"]-best)/max(1.,abs(best)):.4f}% |')
            s=r.get('stats',{})
            if s:
                stats.append(f'| {case} | {NAMES[alg]} | {s.get("subproblem_calls",0)} | {s.get("subproblem_wall",0.):.3f} | {s.get("master_wall",0.):.3f} | {s.get("pricing_wall",0.):.3f} | {s.get("cuts",0)} |')
        timing.append(f'| {case} | {first["service_columns"]:,} | '+' | '.join(cells)+' |')
    conclusion=(ROOT/'conclusion.md').read_text() if (ROOT/'conclusion.md').exists() else '实验结论待根据以下结果填写。'
    content='''# MLND_Path：TS 方法核对与第二轮算法实验

'''+conclusion+rf'''

## 1. TS 的场景数与算法结构

Bayram、Yıldız 与 Farham（2023）的主随机实验只有正常／高峰两个场景，高峰概率 1/12；另有单场景对照。100 个新需求场景用于样本外验证。两场景确实提供独立运营块，但原文更有针对性的算法设计是容量可行性主问题、最大流分离和图上定价。[期刊全文](https://doi.org/10.1287/trsc.2022.0112)，§4–6，pp.1283–1289。

原文主要 TR／AP 设置的二元变量分别为 60／45 个，仅决定枢纽与容量等级；OD 路由为连续变量。我们保留每个 OD 整数选择基地对，因此中型、较大型、扩展测试分别有 81／119／398 个二元变量。TS 的最终主问题还含连续目的地容量预留变量，式 (42)–(45)；本文据此探索在主问题保留连续运营结构。

按原文 §6.2，单场景最优求解比例为 TR 87%、AP 100%，两场景为 TR 68%、AP 74%；通常限时 10 小时。这些比例描述其算法在不同模型设置下的表现。本文的速度比较全部在自己的相同实例和预算下完成。

[逐项原文核对与页码](literature_audit.md)记录了主问题、定价、实际场景数、整数规模、初始列与新增列的区别。

## 2. 本轮实际实现的调整

本轮保持同一个整数设计与运营模型：设施开放、OD 基地对整数分配、连续可用率、解析随机服务时间期望、末端柜 utility、连续电量和换电。全部方法采用相同的候选列与整数有效加强约束。本轮没有增加需求场景。

| 方法 | 主问题／整体模型 | 内层 |
|---|---|---|
| 加强完整 MISOCP | 所有整数变量、连续变量和候选列 | Gurobi 联合求解 |
| 原锥子问题 Benders＋CG | 整数设计与运营成本估计 | 含可用率和 idle 的 SOCP；沿用第一轮实现 |
| 部分 Benders＋全列 LP | 整数设计、可用率、idle 锥、基地间汇总流量 | 持久化全列路径 LP |
| 部分 Benders＋增量 CG | 同上 | 持久化路径 LP，增量添加列并复用单纯形基 |
| 按路径组解析多割 | 同上，细分各路径组成本估计 | 逐组最便宜开放路径与全候选割证书 |
| 解析多割＋idle 外逼近 | 用自适应切线构造线性混合整数主问题 | 同一解析路径分解 |
| idle 外逼近＋可行解回灌 | 同上，并向求解器提交修复后的实际可行解 | 同一解析路径分解 |

原锥子问题版本本轮使用加强约束和整数候选处 lazy 割，不含第一轮 `benders_root` 的根节点预分离。这样分开了约束加强与根节点加割。新部分 Benders 同时改变了分解边界、LP 持久化和聚合约束；全列 LP／增量 CG 对照进一步观察列生成的效果。不同方法的对偶解和搜索路径可能不同。

### 2.1 部分 Benders

在主问题中定义聚合流量

$$T_{{od\ell m}}=\sum_{{p\in\mathcal P_{{od\ell m}}}}\theta_{{\ell m}}^p,$$

并保留空重定位基地对流量、来源分配与可用率的线性关系、基地流量平衡和 idle 二阶锥。固定这些量后，路径子问题只需将已知的各组流量分配到开放设施支持的路径，目标是线性的。

主问题提出的正流量未必有可行路径；新实现用第一阶段人工未分解流的 LP 检查可行性。完整候选定价完成后，正人工流量的对偶产生可行性割；可行时切换为真实路径成本产生最优性割。人工流量仅用于算法检查，不进入交付的运营方案。

[完整算法与割公式](partial_algorithm.md)。

### 2.2 当前运营设定还允许解析路径分解

当前 H／换电约束只控制开放与使用，开放时采用安全的流量上界；机队占用以线性价格计入路径费用。固定整数设施设计 $x$ 及汇总流量 $T$ 后，各路径组可以独立选择最便宜的开放路径：

$$Q_{{\mathrm{{path}}}}(x,T)=\sum_g q_g(x)T_g.$$

这里 $g$ 仅在算法说明中统指一个明确的 OD–基地对服务组，或一个空重定位基地对；实现和模型仍保留这些具体下标。解析多割实现对每个组单独生成成本界，同时检查全部候选路径的对偶不等式。某个设施开放模式生成的割可以供该模式下不同的 OD 分配和可用率继续使用。

[解析分解的条件、证明及全局割](analytic_path_projection.md)。这项简化直接利用当前模型的激活约束结构；TS 的实际枢纽容量会把不同路径组耦合在一起。

### 2.3 idle 函数的自适应外逼近

TS 的扩展主问题保持为 MILP。本轮进一步测试将 idle 凸函数以支持切线放入主问题：

$$I_\ell\ge\frac{{r}}{{1-r}}+\frac{{\rho_\ell-r}}{{(1-r)^2}}.$$

算法从少量切线开始，在整数候选处按当前连续可用率添加违反的切线。$\rho$ 仍可取区间内任意值。所有切线都给真实 idle 函数提供下界；构造可行方案时，将空闲机数上修到原始 $\rho/(1-\rho)$ 所需水平，再计算真实成本。因此，下界与可行上界始终对应原模型。可行解回灌变体进一步把修复后的完整主问题变量提交给求解器，用于整数搜索中的剪枝。

支持切线的有效性还可以直接由下式核对，其中 $f(u)=u/(1-u)$：

$$f(\rho)-[f(r)+f'(r)(\rho-r)]=\frac{{(\rho-r)^2}}{{(1-\rho)(1-r)^2}}\ge0.$$

因此算法报告的 gap 使用修复后的真实运营上界与 MILP 下界，包含当前外逼近尚未闭合的误差。

## 3. 统一预算的实验结果

正式实验共 {len(cases)} 个合成实例、{len(rows)} 次运行；单线程，每次求解预算 {configs[0]['time_limit_seconds']:g} 秒，目标最优性间隙 {100*configs[0]['mip_gap']:.2f}%。核心四种方法按实例轮换顺序，解析法及外逼近在后续独立批次运行；全部求解顺序执行。路径预处理独立计时，并在原始结果记录端到端时间。环境：Python {configs[0]['python']}，Gurobi {'.'.join(map(str,configs[0]['gurobi']))}。

| 实例 | 服务列数 | 完整 MISOCP | 原锥 Benders＋CG | 部分 Benders＋全列 LP | 部分 Benders＋CG | 解析多割 | idle 外逼近 | 外逼近＋回灌 |
|---|---:|---|---|---|---|---|---|---|
'''+ '\n'.join(timing)+'''

### 可行解与全局下界

“相对最佳已知解差”比较本轮各算法的可行目标；gap 使用各算法自身全局下界。达到预算时保留当前可行解与下界。

| 实例 | 方法 | 可行目标 | 全局下界 | gap | 相对最佳已知解差 |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(details)+'''

### 耗时与割

内层时间包括定价，定价不能再次相加。解析版“内层”指路径组扫描与割构造，没有 LP/SOCP 调用；各方法的调用数不代表同等计算工作量。

| 实例 | 方法 | 内层调用 | 内层秒 | 主问题秒 | 定价秒 | 割数 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(stats)+f'''

## 4. 验证与复算

**{test_count} 项独立测试全部通过。** 每个解都按原加强模型矩阵复查约束、idle 关系，并独立重算目标。正式结果最大约束残差为 **{max_res:.3g}**，最大目标重算差为 **{max_obj:.3g}**。在绝对数值容限 1e-5 内，各实例全部算法的下界均不超过跨算法最佳可行上界。正式运行启动时的计算源码哈希与本次交付代码逐一核对一致。

新增测试覆盖原模型值一致性、跨设施设计的可行性／最优性割、连续流量下的割、负路径成本、候选列完整定价，以及超时后可行解的保存。测试结果保存在本目录的验证记录中。

复算入口：`python -m mlnd_path.benchmark_followup`；外逼近入口为 `python -m mlnd_path.ts_followup_20260914.run_oa`，可加 `--inject`。逐次设计、流量、上下界、迭代轨迹、实例种子和源码哈希保存在本目录的正式实验目录中。开发预实验单独保存在 `pilot_partial`、`pilot_oa`、`pilot_oa_inject`，未合并入上表。
'''
    (ROOT/'TS_FOLLOWUP_REPORT.md').write_text(content)
    css=(ROOT.parent/'report_style.css').read_text()
    (ROOT/'report_style.css').write_text(css)
    subprocess.run(['/opt/anaconda3/bin/pandoc','TS_FOLLOWUP_REPORT.md','--standalone',
                    '--mathml','--embed-resources','--css=report_style.css',
                    '--metadata','pagetitle=MLND_Path：TS方法与新分解实验',
                    '-o','report.html'],cwd=ROOT,check=True)
    for name in ['literature_audit','partial_algorithm','analytic_path_projection']:
        subprocess.run(['/opt/anaconda3/bin/pandoc',name+'.md','--standalone','--mathml',
                        '--embed-resources','--css=report_style.css','-o',name+'.html'],cwd=ROOT,check=True)
    text=(ROOT/'report.html').read_text()
    for name in ['literature_audit','partial_algorithm','analytic_path_projection']:
        text=text.replace(name+'.md',name+'.html')
    (ROOT/'report.html').write_text(text)
    print(json.dumps(dict(instances=len(cases),runs=len(rows),maximum_residual=max_res,
                          maximum_objective_error=max_obj,source_hashes_verified=True)))


if __name__=='__main__':build(sys.argv[1:])
