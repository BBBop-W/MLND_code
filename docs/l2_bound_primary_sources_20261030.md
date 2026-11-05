# 当前 MLND baseline 的上下界改进：两条有条件的研究路线

核对日期：2026-10-30（Asia/Shanghai）。本稿是背景研究，未修改求解代码、运行优化或改动实验。对象为 `baseline-20261029`：固定任务率、K2、origin 分配、aggregate flow，固定整数设计后的运营 recourse 为 SOCP。**建议优先研究小型联合运营核心以加强 LB，同时把现有上界搜索改成资源块邻域与精确 recourse 结合；两者均没有本轮实测提升证据。**

## 1. 先排除已经做过的方向

已核对 [当前 baseline](current_baseline.md)、[20261028 一手文献](l2_cut_primary_sources_20261028.md)、[资源模型缺口](l2_cut_model_gap_20261028.md)、[面割构造](l2_face_cut_construction_20261028.md)、[价格优化数学稿](l2_next_cut_math_20261028.md)及后续实现稿。资源 LP 已包含共享可用率、联合路径流选择与空载平衡；队列 EPI、价格搜索、分数点分离及目标精化已实现。不能再次把“加入共享 rho/空返平衡”当成新方案。

上界侧也已有受限列池 MISOCP、local branching、固定 assignment 的 polish，以及候选设计的 recourse 评估；见 [restricted_search](../cpp/src/benders.cpp) 和 [solve_restricted_heuristic](../cpp/src/heuristics.cpp)。当前 local branching 使用 facility/capacity/assignment 坐标的距离行，不能据此说已经实现了跨这些块与航线的资源耦合 ALNS；但也不能把普通局部搜索重新命名为新算法。以下建议只针对这些现有机制的差异。

## 2. 下界：保留小型联合运营核心，而非只累积独立成本下界

**文献机制。** Partial Benders 将部分 recourse 变量和关联约束留在 master，使其直接看见运营结构。Crainic–Hewitt–Rei 的作者稿 §3.1、式(10)–(15)明确保留部分场景；正式后续论文为 Crainic–Hewitt–Maggioni–Rei（2021）。它们的场景分离结构不能直接等同于本项目耦合的 OD 任务。[作者原稿](https://www.cirrelt.ca/documentstravail/cirrelt-2014-13.pdf)、[正式论文](https://pubsonline.informs.org/doi/10.1287/trsc.2020.1022)。

更接近当前确定性网络问题的是 Bélières–Hewitt–Jozefowiez–Semet（2022）的聚合商品 master：§4.2 / Theorem 4.1 用原解到聚合解的映射证明它是松弛。Theorem 4.3 支持在既有分组内部细分而逐步加强；**任意增加分组数量并不保证更强**。其商品可替代性和线性成本条件不是 MLND 已满足的结论。[作者预印本 §4.2–4.4](https://arxiv.org/pdf/2009.14628)、[正式论文](https://doi.org/10.1016/j.ejor.2021.07.056)。

**本模型的建议迁移。** 设计变量仍记为 (x)，完整运营变量记为 (v)，固定设计的值函数为 (Q(x))。选取少量真正耦合的运营量 (u=P(v))，构造主问题中的外松弛 (mathcal R_{
m core}) 及成本下界 (ell(x,u))。首版可考察两个基地的可用率与精确队列 epigraph，配合少数任务组、柜占用/航线容量和返回平衡的联合约束；其他路径流必须有安全的聚合或消元表示。候选核心应回答“已有价格行包络不能同时表达哪组耦合”，而非重复已有的单行资源下界。

需要逐条证明：

\[
\forall (x,v)\in\mathcal F,\qquad
(x,P(v))\in\mathcal R_{\rm core},\qquad
\ell(x,P(v))\le q(v).
\]

于是保留既有全部有效割，并增加

\[
(x,u)\in\mathcal R_{\rm core},\qquad \theta\ge\ell(x,u)
\]

不会排除真实设计—成本组合，所得 master 的认证 best bound 才能作为全局 LB。这是针对 MLND 的直接投影论证，**不是上述 MILP 论文已证明当前 SOCP 核心有效**。还没有构造并验证具体 (mathcal R_{
m core})，因此当前状态为研究方案。

必须保留的边界：

- 保留有限条真实路径、把其他路径直接设为零，通常得到的是限制模型，不能作为 LB 核心。被省略路径必须有覆盖完整域的松弛表示，或保持完整定价认证。
- 单/双任务率、第二单出现次数、AA/self-pair、基地返回及共享资源系数均沿原模型；不能把相同 head 下的不同任务率随意合并，更不能改变原订单归属。
- 若 (ell) 已估计完整运营成本，与普通 recourse 割应同时保留，即要求 (theta) 高于二者；不能把两个完整成本下界相加。只有先证明成本分解，才可使用“核心成本＋剩余 epigraph”。
- 新核心若只是完整连续松弛的另一表达，不能仅凭扩展变量就宣称跨越其连续最优值；它可能改善有限割 master 或求解器对整数结构的利用。严格增强、正的最终 LB 增量和总时间收益是三件不同的事。
- 分组细化要保留旧核心/旧割或建立嵌套投影；更大的 master 也可能更慢。主问题只加少量变量，不代表完整域系数构造便宜。
- 若把精确队列锥约束留在 master，主问题会变为 MISOCP；若沿用当前 MILP 主问题及分数 LP 接口，则需采用已证明的外逼近，并相应核对 LP 复制与认证范围。不能把锥约束默认为已被线性行接口完整保留。

**为何不优先重新启动 disjunctive/Lagrangian Benders。** 它们仍是合理备选，但 [已有面割稿](l2_face_cut_construction_20261028.md) 已覆盖三态分支和首次失配提升。Rahmaniani et al.（2020）Proposition 1 / Theorem 1 允许复制设计并保留部分整数性，收益依赖内层离散间隙；原文也指出整数紧锚点的限制。[作者原稿 §2.1](https://www.cirrelt.ca/documentstravail/cirrelt-2019-27.pdf)。在当前 SOCP 中，只能使用弱对偶迁移：

\[
\phi_J(\beta)=\inf\{q(v)-\beta^\top z:(z,v)\in\mathcal R,\ z_J\text{整数}\},
\quad \underline\phi_J\le\phi_J
\ \Longrightarrow\ \theta\ge\underline\phi_J+\beta^\top x.
\]

这里 (mathcal R) 必须覆盖原完整路径域；内层可行目标是上界，受限路径 MIP 的 best bound 也不是该完整域截距。对多个析取分支，所有分支都要有合法下界，不能丢弃未求完分支。先得到一个可压缩、可证明的联合核心，再在其中保留少量整数性，比直接展开全域整数内层更可控；其收益仍待验证。

## 3. 上界：资源块 RINS/ALNS，接现有 SOCP recourse

**文献机制。** Danna–Rothberg–Le Pape（2005）的 RINS 用 incumbent 与连续松弛引导子 MIP；官方 SCIP 实现 `determineFixings` 固定两解取值一致的整数变量，留下其余变量搜索。CPLEX 官方说明该子 MIP 受节点预算限制。[原论文出版页](https://link.springer.com/article/10.1007/s10107-004-0518-7)、[SCIP 官方实现](https://www.scipopt.org/doc/html/heur__rins_8c_source.php)、[CPLEX 官方说明](https://www.ibm.com/docs/en/icos/22.1.2?topic=heuristics-relaxation-induced-neighborhood-search-rins-heuristic)。

Ropke–Pisinger（2006）的 ALNS 让多个子启发式按历史表现竞争调用机会。可迁移的是这种调度原则；其 PDPTW 路线破坏/插入操作并不直接适用于固定任务率的 MLND。[作者机构记录与稿件入口](https://orbit.dtu.dk/en/publications/an-adaptive-large-neighborhood-search-heuristic-for-the-pickup-an/)、[原论文](https://pubsonline.informs.org/doi/10.1287/trsc.1050.0135)。

**建议只扩展现有 primal_search 接口。**

1. 由 incumbent 的实际流量与当前 master 分数解选一个资源块，例如“一个柜点/航线及依赖它的 head 分配”，联合释放相关容量级别、完整三态分配组、开关航线；不只允许若干独立二元位翻转。基地转换时把必要返航弧纳入同块。其余一致的整数组可作 RINS 式固定。
2. 在原模型约束下用已有列池做有预算的邻域 MISOCP，复用 incumbent 作为局部可行初解；得到少量不同设计后，以现有完整域 CG/SOCP 评估并补充路径。当前 master LP 只是指导点；这属于 **RINS-inspired** 扩展，不能称为完整原模型 LP 的经典 RINS。
3. 多个候选块/固定方式共享既有启发式总预算，以“通过原始模型验证的 UB 改善 / 实际累计搜索时间”更新调用权重，并保留少量探索。这里按时间计分是本项目建议，不是原 ALNS 论文的原样公式；没有证据时不先增加总预算。

**有效性与 exact recourse 的准确含义。** 任意邻域候选只要满足全部原物理/队列/任务约束、成本按原模型重算，就是合法 UB；即使它来自受限列池，也不要求证明邻域最优。对一个固定设计，只有完整域定价完成、原始与对偶证书达到要求，才称其 recourse 已精确求解（在声明数值容差内）。未完成定价但已验证的可行运营解仍可改进 UB，只是不能称为精确设计评价。由邻域求得的 best bound、固定设计 recourse 下界、或 master 的虚假 (theta) 均不得更新原问题 global LB。

邻域冻结是临时搜索限制，不是全局变量固定；邻域无解或超时也不能生成原问题不可行结论。近似相等容差只能决定如何选择邻域，不能用来放宽最终可行性。容量 one-hot 与 head 三态要按整组处理；第二单出现不要求其自身 head 分配开启。若要顺便生成普通 Benders 割，仍须通过既有完整域对偶认证路径，不能把启发式目标当截距。

这一方向的新增点是**联合资源块、松弛引导的固定以及有反馈的调度**，其余候选校验和精确 recourse 应复用现有可靠部分。仅打开 CPLEX 内置 RINS 参数，未必能改变外层 Benders 看到的运营信息；本稿不声称该参数对当前 MISOCP 一定有效。

## 4. 优先顺序与证据等级

| 路线 | 建议优先级 | 已有依据 | 仍需完成的关键事项 |
|---|---|---|---|
| 小型联合运营核心 / partial Benders | 下界主线 | 原论文给保留结构/聚合外松弛机制；本稿给投影有效性的充分条件 | 写出具体核心，证明覆盖完整域且不重复算成本；确认它比当前割包络增加了信息 |
| 资源块 RINS/ALNS＋原 SOCP recourse | 上界辅助线 | 原论文/官方代码给邻域与调度机制；现代码已有可复用候选验证接口 | 明确释放块、候选去重与共享预算；观察有效 UB 改善，而非仅邻域目标降低 |

上述均为研究建议。没有新实验，因此不宣称提高了任何实例的 LB、UB、gap 或速度；已有 20261028/20261029 实测只用于识别哪些机制已经实现，不能作为这两条新方案的性能证据。

来源阅读范围：已读 partial-decomposition 作者稿 §3.1、Meta-PBD 作者稿 §4.2–4.4/相关证明、BDD 作者稿 Proposition 1/Theorem 1、SCIP 官方 RINS 固定逻辑和 CPLEX 官方说明。RINS 原论文与 ALNS 本次核对到出版社/作者机构的摘要与元数据，未把未取得的全文细节当作已精读事实；核心建议所需的 RINS 固定行为另外由官方源码直接确认。全部网络引用为原始论文、作者机构稿或官方文档/源码，未以综述代替原文。
