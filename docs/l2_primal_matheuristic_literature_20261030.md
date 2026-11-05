# MLND 初解与上界搜索：元启发式是否值得做

调研日期：2026-10-30（Asia/Shanghai）。本稿完成一手文献读取和针对当前模型的设计推导，没有改算法、运行优化器或新增消融；H40 不在范围内。下列文献证明各自原问题上的算法机制，不证明迁移后一定改善 MLND。

**建议研究“多起点构造 + 资源块大邻域 + 原模型求解器修复”的数学规划混合启发式。先保证从全骑手解能够产生非空、分支完整的候选，再考虑 ALNS 的自适应权重。** 当前问题不是缺一个元启发式名称，而是构造空间、联合改动尺度和昂贵评价之间的配合。独立的 GA/SA 路线搜索不作为首选；普通 RINS 也不能替代激活一个新服务网络的算子。

## 1. 六个核心一手来源与实际读到的内容

| 原始来源 | 本次读取范围 | 已核实机制 | 对本项目的迁移与限制 |
|---|---|---|---|
| Ropke, S.; Pisinger, D. (2006), *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows*, Transportation Science 40(4), 455–472. [DOI](https://doi.org/10.1287/trsc.1050.0135)；[作者机构页面](https://orbit.dtu.dk/en/publications/an-adaptive-large-neighborhood-search-heuristic-for-the-pickup-an/) | DTU 作者机构摘要、出版信息及 Crossref 出版社元数据；全文链接本次未能读取 | 多个子启发式竞争，调用频率根据历史表现更新；原问题为带时间窗的取送路径问题 | 可迁移的是算子组合和反馈调度；不能把订单删除/重新插入的路线算法照搬到固定任务率和共同 rho 模型，也不宣称本次已核对原文所有接受规则细节 |
| Hendel, G. (2018), *Adaptive Large Neighborhood Search for Mixed Integer Programming*, ZIB Report 18-60. [作者稿页面](https://optimization-online.org/2018/12/6992/)；[全文](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf) | 全文，重点 §2–3、Table 1 | 在 SCIP 中组合八种 solver neighborhood；用 bandit 选择、受限子 MIP、动态固定比例/节点预算控制成本。变量—约束图用于保持相关变量块；RINS、RENS、Local Branching 的前提不同 | 最直接支持“少数资源块算子 + 有预算的求解器修复”。先有能产生改善的算子，适应性调度才有作用。原证据来自 MILP，不能直接声称当前 CPLEX MISOCP 已提供同样能力 |
| Fischetti, M.; Lodi, A. (2003), *Local branching*, Mathematical Programming 98, 23–47. [原出版页](https://link.springer.com/article/10.1007/s10107-003-0395-5) | 出版社摘要；二元距离邻域的公式另经 Hendel §2.2 核对 | 外层选择邻域，通用 MIP 求解器探索受限子域，争取早期好解 | MLND 应按完整 head 三态、容量 one-hot 和关联航线构成资源块；固定一个很小的普通二元 Hamming 半径可能不允许完整开启服务网络。现有 local branching 开关不等于已经有这种资源块邻域 |
| Danna, E.; Rothberg, E.; Le Pape, C. (2005), *Exploring relaxation induced neighborhoods to improve MIP solutions*, Mathematical Programming 102(1), 71–90. [原出版页](https://link.springer.com/article/10.1007/s10107-004-0518-7) | 出版社摘要与元数据；固定规则另经 Hendel §2.1 / Table 1 核对。出版页先于卷期于 2004 在线发布 | 利用 incumbent 和连续松弛构造邻域，再求解子 MIP；经典固定规则是固定两解一致的整数变量 | 当前 Benders master LP 可作启发式指导，但不是完整原模型连续松弛。应称 RINS-inspired；不能将其一致坐标永久全局固定。从全骑手解出发时必须主动释放一整组潜在服务资产 |
| Hewitt, M.; Nemhauser, G.; Savelsbergh, M. W. P. (2013), *Branch-and-Price Guided Search for Integer Programs with an Application to the Multicommodity Fixed-Charge Network Flow Problem*, INFORMS Journal on Computing 25(2), 302–316. [DOI](https://doi.org/10.1287/ijoc.1120.0503)；[作者 2012 稿](https://optimization-online.org/wp-content/uploads/2012/02/3360.pdf) | 作者全文，重点 §1–3；正式出版信息经 Crossref 核对 | 利用网络问题结构定义较小整数限制问题，并由列生成引导；结合 incumbent 周围的局部搜索 | 与 MLND 的固定开设成本、路径列池及受限 MIP 接缝更接近。可借用结构化限制和解池；该论文完整 branch-and-price 的全局保证不能移植给 MLND 的受限路径启发式 |
| Bonami, P.; Cornuéjols, G.; Lodi, A.; Margot, F. (2006), *A Feasibility Pump for Mixed Integer Nonlinear Programs*, IBM Research Report RC23862. [作者稿页面](https://optimization-online.org/2006/02/1332/)；[全文](https://optimization-online.org/wp-content/uploads/2006/02/1332.pdf) | 全文的摘要与初始算法、改进解讨论 | 面向凸 MINLP，在 NLP 与 MILP 之间交替寻找整数可行解；也讨论改善已有解和结合外逼近 | MLND 已有全骑手整数可行解，因此单纯“找可行解”不是缺口。可以研究带成本改进门槛的变体，但优先级低于能联合激活设施和完整任务组的构造/修复 |

只使用上述作者稿、作者机构或原出版方页面作为算法机制来源。公开检索工具当时连接失败，本次通过只读 HTTPS 读取上述公开来源；全文未读到的部分明确标注，没有把二手综述当原文。

## 2. 为什么更适合数学规划混合启发式

本节为针对 MLND 的推导，而非文献已验证结论。

战略部分包含设施、容量档位、head 的未分配/基地 1/基地 2 状态和航线，运营部分含固定正率 single/pair 分支、共享 rho、队列、容量与返回平衡。按路径排列做随机变异不能自动保持这些约束；将所有连续运营量也纳入随机编码，会重复现有锥求解器擅长的工作。更自然的分工是让搜索决定“释放哪些相关设计和哪些路径候选”，原模型求解器负责连续运营和局部整数修复。这与 Local Branching 的战略/战术分工、Hendel 的有预算邻域和 Hewitt 的网络结构限制相符；具体应用仍须验证。[Local Branching](https://link.springer.com/article/10.1007/s10107-003-0395-5)、[Hendel §3](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf)、[Hewitt 等 §1–3](https://optimization-online.org/wp-content/uploads/2012/02/3360.pdf)。

**真正的起点应是构造多个“可运营的资产—任务组合”，不要求每一步单独改善当前成本。** 举例，仅作为固定成本结构说明：两组任务各自节省 6，合开共享设施需 10，则每组单开净收益为 −4，合开净收益为 2。只允许正的单步增量会在空网络停止，增加随机种子也无效；要允许成组激活或有限深度前瞻。该例不是已测 MLND 的数值结论。

完整分支覆盖是必要的候选域构造条件。不能通过删除低收益正率分支、改写预处理任务率或让不同分支使用不同的 head/base 决策，使局部搜索容易。最终保留的可行方案必须仍满足原模型全部约束；多起点只是搜索路径的不同，不是需求模型的不同。

## 3. 建议分两步实现，不先做一个大而全的 ALNS

### 第一步：多起点激活构造，解决 all-courier 自启动

建议保留 3–5 种真正不同的构造方向，例如共享柜/航线簇、同基地的任务簇、两基地互补返回，以及连续松弛建议的资产簇。每种方向先构造一个足以覆盖所选 head **全部正率分支**的路径并集，再短时求解受限原 MISOCP。允许少数资产或 head 组合先被联合加入，然后按完整成本决定是否接受；评分用于排序，不能把单步非正评分当成该组合永远无价值的证明。

资源块大小应随覆盖需要与求解代价调整，不能由固定“最多 6 个柜”永久截断。必要时先放宽候选域覆盖，再在受限模型内由开设成本选择实际使用的柜；不是要求预先打开全部候选柜。构造器必须在 `served=0` 时可运行，且其触发不应等待某个完整域 recourse 收敛后才得到第一次机会。

这是 GRASP 风格多起点/受限候选集合的设计选择，但本轮不把某个现成 GRASP 实现或其性能作为已核实依据。Resende–Werneck 的 *A hybrid multistart heuristic for the uncapacitated facility location problem*（2006）书目信息已由[作者出版列表](https://mauricio.resende.info/papers.html)和[DOI](https://doi.org/10.1016/j.ejor.2005.02.046)核对；本次未读到其全文，不据此细述或套用其具体算法。当前优先级由 MLND 的构造空域诊断决定，不由“GRASP”名称决定。

### 第二步：资源块 VNS/LNS，再按证据升级为 ALNS

初版可轮换以下少数算子，并随停滞逐步扩大相关块。这是建议的 variable-neighborhood 策略，不声称已证明优于 ALNS。

| 算子 | 联合释放内容 | 用途 |
|---|---|---|
| 激活一个新簇 | 一组 head 的完整三态、全部正率分支路径、候选柜/容量/航线及必要返回 | 从全骑手或稀疏无人机网络产生新服务，而非只微调已有解 |
| 关闭/替换资产 | 高固定成本或低利用率的柜/航线及依赖它的 head；新旧路径资产同时释放 | 实现共享、替换和真实关站收益 |
| 基地迁移 | 同基地/分配分歧 head，关联空载和跨基地返回资源 | 改变共同 rho、机队和基地平衡的组合 |
| 拥堵缓解 | 同一紧柜容量或瓶颈航线影响的 head 与替代柜 | 避免只改一个任务却被其余任务的共享资源状态锁死 |
| 精英重组 | 两个已验证好解相异的完整设计组和两者正流路径并集 | 有多个不同好解以后，才研究 path-relinking 或 crossover 风格操作 |

没有多个不同有效解时不启动精英重组；没有可靠 LP 指导点时不用 RINS 式固定；二元距离按组计量，避免容量或分配 one-hot 编码人为扭曲距离。这些前提与 Hendel 对不同邻域的输入条件区分一致。[Hendel Table 1](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf)。

只有这些算子确实有不同的成功率/耗时表现，才增加自适应权重或 bandit。建议奖励由“通过原验证的 UB 改善、得到不同可行解的价值、真实秒数”组成，保留探索概率；用真实秒数和 MLND 的组级资源调整是本项目建议，不能归为原始 ALNS 论文的原样实现。[ALNS 原摘要](https://orbit.dtu.dk/en/publications/an-adaptive-large-neighborhood-search-heuristic-for-the-pickup-an/)、[Hendel §3](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf)。

## 4. 昂贵评价必须单独解决

建议采用三级流程，只有第一层是启发式评分：

1. **廉价筛选。** 对共享资产增量成本、所有分支的路径可达性、容量和平衡做快速检查；缓存设计/资源块签名，去掉相同候选。可以使用已有成本/价格代理排序，但代理低估不产生合法 UB，高估也不能未经证明就否决一切潜在组合。
2. **有预算的原模型修复。** 只在候选路径并集加全部 incumbent 正流支持上求解原 SOCP/MISOCP。得到通过原始可行性和完整成本检查的整数设计—运营方案即可更新 UB，**不必等待固定设计完整域 CG 最优性证书**。任何正率分支、queue、容量或平衡约束都不能省略。限制路径域得到的合法可行解对原问题仍可行；限制域的 best bound 不得进入全局 LB。
3. **精英精修。** 只对少量真正有竞争力或反复出现的设计继续完整域定价/CG，改善路径及运营分配；预留中断点、复用已有列和可行解。没有完整定价和相应证书时，只称可行评价或受限域评价，不能称已算出精确的设计值函数。

Hendel 明确把 LNS 视作强但昂贵的启发式，并用子问题规模和预算约束其耗时；Hewitt 的论文也将高质量解搜索与较小结构化整数限制问题相连。它们支持“控制每次修复成本”这个方向，**并未直接给出以上 MLND 的三级评价算法**。[Hendel §1、§3](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf)、[Hewitt 等 §1](https://optimization-online.org/wp-content/uploads/2012/02/3360.pdf)。

普通 feasibility pump 并非完全不能改善目标；原文明确讨论改进已有解。但 all-courier 已可行时，优先目标应是得到低成本、正服务的组合。单纯减少整数违反量、追求第一份可行解，无法解释或保证这种提升。若未来使用 pump，应包含成本改善要求并核对原锥约束，不能只把 MILP 近似的可行点当作原解。[Bonami 等作者稿](https://optimization-online.org/wp-content/uploads/2006/02/1332.pdf)。

## 5. 应如何判断是否值得继续

首要判据是首次有效无人机解的出现时间、初始 UB、后续相同总预算下的最终 UB，以及原模型验证；同时单列构造、修复、精修和失败调用的实际时间。H20 若仍返回空候选集合，先修构造域；若有完整候选而修复常超时，先减小或重组受限模型；若能稳定得到不同可行解但容易停滞，再投入 ALNS 调度或精英重组。这里不要求新增消融矩阵，也不把本次调研当成实施/收益证据。

结论：**值得把下一轮主要精力转到上界自启动与便宜、可行的评价上，再用资源块 LNS/VNS 组织搜索。** ALNS 是可选的调度层，不是绕过缺分支、候选域过小或慢 recourse 的办法。仍保留独立的精确 Benders 下界与终止证书；启发式提高 UB，不替代全局最优证明。
