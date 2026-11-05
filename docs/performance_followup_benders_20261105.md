# 下一轮 Benders 加速候选：减少重复认证，再改善设计搜索

2026-11-05。基于当前 `accelerated-20261105` 工作树的研究建议，**未实现、未做性能试验**。仅检查源码及公开一手资料，未读取 `outputs/results`。本文不重复已经实现的图缓存、同设计 RMP 持久化、平滑定价、割池去重/休眠、原始邻域支持补齐与分组距离。

最值得先做的是**设计级证书记忆**；其次是保留独立全局下界的 **level/proximity 候选主问题**。联合面界属于更有理论潜力但较贵的加强方向；single-tree 应放在这些机制与超时协议稳定之后。

| 新方向 | 当前最接近的机制 | 新增内容 | 优先级与前置证据 |
|---|---|---|---|
| 设计级区间/证书记忆 | 当前待续设计保留 RMP；割池保留仿射割 | A→B→A 时复用已认证运营区间、原始点与证书 | P1；先统计重复设计及可避免 CG 次数 |
| 双主问题的 level/proximity 选点 | root in–out、内部点割、预算调度、原始分组邻域 | 改变下一次**运营认证**的设计选取，独立保留全域主问题 | P2；设计振荡/同类重复且认证昂贵时 |
| 首单—容量联合面界 | 单首单三态面界；固定资源价格的双首单队列割 | 联合固定少量首单和容量档位，保留共享资源再求条件下界 | P2 研究原型；先找严格额外违反的小反例 |
| Single-tree branch-and-Benders-cut | 外层反复求解同一 master solver | 在一个整数搜索过程中分离并认证候选 | P3；先确认外部 master 重启/重复搜索成本 |

下文记原问题为 \(\min_{x\in X}\{c^\top x+Q(x)\}\)，其中 \(X\) 是有限战略设计域，\(Q\) 是同一固定任务、完整物理路径域上的连续运营 SOCP，\(\theta\) 是运营成本 epigraph。

## 1. 设计级证书记忆：先避免重新证明已经知道的结论

**代码依据。** 当前 `integer_designs` 记录重复，但 `repeated_design` 后仍进入 `subproblem`；新 `RecourseModelState` 只保存一个同设计模型，且明确不存 Cut。割池保留的是割而非每个设计的运营值区间。因此这个方向尚未实现。[重复设计路径](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:3865)、[当前持久状态](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:68)、[割池](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/certified_cut_pool.hpp:9)

按“数学输入身份＋完整战略设计”保存

\[
\ell_z\le Q(z)\le u_z,
\qquad (\text{有效割、完整定价证书、已验证原始点、认证容差}).
\]

这里 \(\ell_z\) 只来自全域有效证书，\(u_z\) 只来自原模型可行点；缓存不要求区间已经闭合。再次遇到 \(z\) 时按当前 \(\theta\) 决策：

1. 若 \(\theta<\ell_z-\varepsilon\)，恢复/加入相应有效割并重新求 master。**不能只跳过该设计**，否则 master 可能无限返回同一点。
2. 若已有可行运营点且 \(\theta\ge u_z\)，该候选的 epigraph 已被可行点证实；可复用其原始点。数值实现仍采用现有可行性/成本核验和容差规则。
3. 若 \(\theta\) 处于未闭合区间内且设计仍可能改善 incumbent，继续 CG；旧的弱割不等于已求出 \(Q(z)\)。只有 \(u_z-\ell_z\) 满足当前认证标准，才可复用“运营最优”结论。

若只存到可靠的点下界而未保留可恢复仿射割，可以使用下列**自行推导的整数点割**。设 \(L\le Q(x)\) 对全部设计成立、\(\ell_z\ge L\)，并记二值 Hamming 距离为 \(\Delta(x,z)\)，则

\[
\theta\ge L+(\ell_z-L)[1-\Delta(x,z)].
\]

在 \(x=z\) 时右侧等于 \(\ell_z\)；其他整数设计有 \(\Delta\ge1\)，右侧不超过 \(L\)，故有效。其作用是恢复点信息，通常弱于已有普通 Benders 割，**不是新增强割的首选**。Laporte–Louveaux 的原论文针对二值第一阶段、完全补救提出有限整数 L-shaped 方法；这里借用有限离散设计的思路，不将其整数运营模型定理直接移植到本 SOCP。该论文本次只读到出版社摘要，以上有效性另行给出证明。[原论文摘要](https://www.sciencedirect.com/science/article/pii/016763779390002X)

**边界。** 不用可行 UB 构造下界割；不能凭“评估过”添加 no-good 排除设计。若以后排除已证明不能改善的设计，必须另外维护被排除域的证明，剩余 master 的 bound 不能无条件当作原域 LB。缓存键必须覆盖任务率、物理域、目标/界、结构加强与条件化语义；目录仅追加已属原域的路径不使证书失效，但换模型必须失效。容差收紧时旧区间需要重检。可限制缓存数量，淘汰记录只损失速度。

**最小验证。** 小例强制 A→B→A：分别测试 \(\theta\) 低于缓存 LB、位于未闭合区间、超过可行 UB，以及割休眠后返回 A。验证前者恢复割并 reopt、中间情形续算、模型改变不命中；统计避免的完整定价/锥求解次数。若设计几乎不重复，收益会很小。

## 2. 双主问题：稳定选点与全局证明分开

当前整数 master 按运营 epigraph 下包络求候选，已有预算调整和目标界强化；分组距离只用于 `solve_restricted_heuristic`，CG 平滑作用于行价格。因此，把距离用于**下一次 Benders 认证点**仍是新方向。[master 求解](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:3527)、[现有原始分组邻域](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:535)

保留原全域 master \(\min c^\top x+\theta\) 负责 LB；另外用相同有效割建立候选问题，例如

\[
\min d_G(x,\bar x)
\quad\text{s.t. }x\in X,\quad
\theta\ge a_i+b_i^\top x\ (i\in\mathcal C),\quad
c^\top x+\theta\le LB+\lambda(UB-LB),\quad0<\lambda<1.
\]

\(d_G\) 可复用首单三态/容量档位的线性分组距离；亦可加有限信赖域。按实际认证收益调整中心和半径，周期性返回不受限制的 master，避免长期困在局部候选域。无需为了距离引入 MIQP。

**证明边界。** 距离目标、加罚目标、信赖域中的最优成本均不是原问题全局 LB。首版只允许未经正则化的全域 master 更新 LB；候选问题超时/不可行就回退，不据此宣称原问题不可行。离散 \(X\) 上 \(\ell_\infty\) 半径小于 1 会冻结全部二值坐标，所以不能机械照搬连续 trust-region 参数。

Linderoth–Wright 全文第 3 节给出了盒形 trust-region、模型更新和实际/预测下降的接受准则，其理论对象是两阶段随机**线性**规划；本文仅借鉴选点思想。其收敛结论不自动覆盖本整数主问题＋SOCP＋预算中断组合。[作者公开全文，第 3 节](https://arxiv.org/pdf/math/0106151)

**收益/失败与验证。** 预期减少跳到差异过大的设计、提高已生成列和有效割对下一次认证的用途；两套 master 的求解成本也可能超过所省运营时间。用一小组设计循环例比较“原候选”和“level候选”，保持相同总认证预算；每轮同时保存全域 bound 与候选目标，检查前者不超过原模型可行 UB，并检查局部无解后确实回到全域。当前 root in–out 已提供部分稳定性，不能把其已有收益计入新方法。

## 3. 联合面界：把容量状态与共享 \(\rho\) 留在同一个认证问题里

**不要重复的内容。** 已有 `shared_rho_joint`、队列 perspective、单首单三态面界；`QueuePolymatroidCuts::separate_joint` 已对两个首单的 9 个三态组合做联合凸化，但其值来自**一个预先固定的完整资源价格 profile**，再给其他首单加安全惩罚。仅把“两首单联合”再包装一次不是新贡献。[共同可用率](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:396)、[三态面界](/Users/songtaowang/Documents/MLND_code/cpp/src/design_face_cuts.cpp:42)、[已有联合队列割](/Users/songtaowang/Documents/MLND_code/cpp/src/queue_polymatroid.cpp:506)

**真正新增的最小单元。** 选两个经常共享瓶颈的首单和一个柜。以两个起点时的三态（不启用／起点 0／起点 1）为例，联合状态为 \(\sigma\in\{0,1,2\}^2\times\{0,\ldots,K_h\}\)；更多起点必须相应扩展状态表。逐状态固定这些设计变量，其他设计仍连续自由；保留原来的共同 \(\rho_l\)、队列锥、柜资源、航线和空载平衡，求完整路径域上的**总成本条件下界** \(L_\sigma\)。尤其保留实际分支流量等式 \(\sum_{p\in P_{rbl}}f_p=\nu_{rb}v_{rl}\)：整数设计下 \(v_{rl}=\alpha_{rl}\rho_l\)，不能把它替换为 \(\nu_{rb}\alpha_{rl}\)。这是 `design_face` 的小块推广，不需要证明按 OD 可分解。[固定任务分支](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:444)

令 \(w_\sigma\) 为各首单状态与柜容量状态的**分组 one-hot 编码拼接**，因而 \(w(x)\) 由现有设计变量及其仿射补项直接给出；这里不引入表示完整联合状态的乘积变量。选择仿射函数

\[
\max_{\alpha,\beta}\ \alpha+\beta^\top w^*
\quad\text{s.t. }\alpha+\beta^\top w_\sigma\le L_\sigma\quad\forall\sigma,
\]

便得到 \(c^\top x+\theta\ge\alpha+\beta^\top w(x)\)。理由是每个原整数设计属于某个 \(\sigma\)，真实总成本至少为该状态下界；因此割对所有原整数设计及其凸包有效。**它只描述条件下界表的凸包，不是整个运营模型的精确凸包。** 未完成的状态用已有全局 LB 填充；不能漏掉状态或把受限原始最优值填成下界。不同小块给出的总成本界取最大，不能相加。

若某些旧单首单面界能够写成同一 \(w(x)\) 的仿射函数，并逐状态取最大纳入每个 \(L_\sigma\)，完整解出上述小 LP 后，所得下包络不会弱于这些已纳入表的面界。这不包含依赖其他未编码 head 变量的旧割，也不证明对全部现有 Benders/队列割严格更强。理论上可捕捉“迁址与柜升级必须同时发生”造成的交互；若共享资源不紧或其他自由变量吸收了耦合，可能没有改善。

Wei–Gómez–Küçükyavuz 研究了指示变量组合约束与非可分凸函数的联合凸化，支持关注**组合逻辑与非线性之间的交互**这一研究方向。但其第 2 节对象为 \(f(h^\top\beta)\) 加指示约束，相关引理要求 \(\operatorname{dom}f=\mathbb R\)；不能把它直接当成本模型 \(\rho/(1-\rho)\)、多资源约束的现成凸包定理。以上有限状态割采用独立的逐状态下界证明。[作者全文，第 2 节、定理 1；第 4 节、定理 3](https://optimization-online.org/wp-content/uploads/2020/06/7868.pdf)

**最小验证与停止条件。** 只取“两首单＋一柜＋两档”的完整小例，枚举全部联合状态并验证每条割；找出当前全部割的 LP 解上仍有正违反的实例。比较单位认证秒带来的全局 LB 改善，而非仅比较割本身违反量。状态数量呈乘积增长；若需反复做大量 SOCP 才得到很小增益，应停止。这个方向优先于无结构地增加大量队列 profile，但后于廉价的缓存/扫描合并。

## 4. Single-tree：值得保留的后续架构方向，不是近期一键开关

当前主循环在外部多次 `solver.optimize`，随后做运营 CG；`NativeSolver` 的 incumbent observer 用于记录，而不是完整 recourse lazy-cut 回调。[外部循环](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:3527)、[observer](/Users/songtaowang/Documents/MLND_code/cpp/src/linear_model.cpp:381)

候选方案是保持 **MILP master＋自定义回调＋独立 SOCP/CG worker**：整数候选先查证书缓存，必要时求运营问题；有效仿射割经 lazy constraint 排除低估 epigraph 的候选，分数节点只做有预算的全局有效割分离。Fischetti–Ljubić–Sinnl 的非可分容量选址论文已使用这类设计，并明确区分可限频的分数分离与关系到 incumbent 正确性的整数分离；其全文也区分 LP 基热启动与 barrier 子问题。[作者全文，第 4.5、5.1–5.3 节](https://msinnl.github.io/pdfs/cfl-techreport.pdf)

必须先解决三个边界：

- **CPLEX 路径。** 官方 Benders strategy 文档描述 MILP 分解，不能据此推断自动策略支持本 SOCP＋动态列。官方 generic `CANDIDATE` context 又明确与自动 Benders algorithm 不兼容；本方案是自己实现 branch-and-Benders-cut，不能混开两套机制。[IBM strategy](https://www.ibm.com/docs/SS9UKU_12.10.0/com.ibm.cplex.zos.help/CPLEX/Parameters/topics/BendersStrategy.html)、[IBM candidate context](https://www.ibm.com/docs/en/icos/22.1.2?topic=context-cpx-callbackcontext-candidate)
- **未决候选。** 若 \(\theta<\ell_x\)，可用已证有效割拒绝；若 \(\theta\ge u_x\) 且有验证原始点，epigraph 已可接受；若落在尚未闭合区间中，超时不是“可接受”或“不可行”的证明。首版应中止此次搜索并回退外部续算，不得添加无证明 no-good。现有完整定价、原始/锥对偶修正和数值回退标准全部保留。
- **线程与列注册。** generic callback 信息是线程局部的；共享注册表、缓存和 CutPool 的修改需要专门协议，当前单个持久 RMP 不能直接并发使用。官方 `ilobendersatsp2.cpp` 为每个线程建立带独立 `IloEnv/IloCplex` 的 LP worker；该例证明的是 LP worker 的接线模式，**不证明这里 SOCP＋CG 的线程安全或速度**。[官方示例目录](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-c-examples)、[本机官方示例](/Users/songtaowang/Applications/CPLEX_Studio222/cplex/examples/src/cpp/ilobendersatsp2.cpp:125)、[IBM generic callbacks 多线程规则](https://www.ibm.com/docs/en/icos/22.1.1?topic=callbacks-multithreading-generic)

**最小验证。** 先单线程、关闭额外分数分离、用可完整枚举的小实例对照外部循环；再人为中断 candidate 的 CG，验证不会接受低估的 \(\theta\)，不会把未决点排除。核对最佳可行值、global bound 和终止状态后，再考虑线程复制。预期可能减少重复 master 搜索；若运营认证才是绝对主耗时，长回调会阻塞树搜索并削弱既有 ALNS 时间片安排。

## 来源与阅读范围

以下均为原作者、原出版方或 IBM 官方来源；没有引用其他问题的性能数字作为本项目的加速依据。

| 来源 | 本次实际阅读范围 |
|---|---|
| [Laporte & Louveaux (1993), *The integer L-shaped method for stochastic integer programs with complete recourse*](https://www.sciencedirect.com/science/article/pii/016763779390002X) | 出版社检索返回的摘要和 DOI；页面直开 403，未读全文，未把本文点割公式归因于其未读定理 |
| [Linderoth & Wright，2001 作者预印本；期刊版 2003](https://arxiv.org/pdf/math/0106151) | 全文可读；重点读第 3 节 trust-region、式 (20)–(31)、第 3 节末 LP/QP 对比；不沿用其连续随机规划收敛结论 |
| [Fischetti, Ljubić & Sinnl，容量选址论文作者稿；期刊版 2016](https://msinnl.github.io/pdfs/cfl-techreport.pdf) | 全文可读；重点读第 4.5–4.6、5.1–5.3 节，区分 in–out、便宜松弛分离、线程 worker 与整数候选认证 |
| [Wei, Gómez & Küçükyavuz，2021-06-15 作者稿](https://optimization-online.org/wp-content/uploads/2020/06/7868.pdf) | 全文可读；重点读第 2 节的问题类、引理 1、定理 1 和第 4 节定理 3；未声称适用其未满足的函数域假设 |
| IBM Benders strategy / candidate context / generic callback 多线程规则 / C++ examples（上文链接） | 搜索工具返回的官方正文；部分直开 403。strategy 可读版本为 12.10，candidate 为 22.1.2，多线程规则为 22.1.1；未推断未核验版本的新增能力 |
| 本机 CPLEX 官方 `ilobendersatsp2.cpp`（上文链接） | 已读注释、Worker 独立环境、ThreadUp/ThreadDown、Candidate/Relaxation 与拒绝/加割接线；未运行示例 |

**建议实际顺序：** 与代码层扫描合并/状态复用工作并行，先做设计证书记忆和命中审计；若 master 候选质量仍限制推进，再做双主问题；联合面界只保留能展示额外界强度的小块；single-tree 最后独立原型化。四者均要求原任务率和完整路径域不变，并以相同总预算下的有效 LB、可行 UB 与认证 gap 衡量，而不是只减少某一类求解次数。
