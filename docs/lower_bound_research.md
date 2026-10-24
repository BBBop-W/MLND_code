# 下界加强：方法适用性、证书条件与诊断依据

核查日期：2026-10-21。代码基线是 `MLND_Continuation`，核查开始时 HEAD 为 `0fe23717de2223b0f2a21634b2d9c9db8a97d464`，工作区存在未提交修改；该提交号不代表工作区内容的完整快照。本页记录方法依据与验收条件，不代表下文所有研究方向已经实现。实现与参数以 [主问题加强](master_strengthening.md)、实际代码和结果文件为准。

## 1. 先明确我们求解的数学对象

令 \(x\in X\subseteq\{0,1\}^n\) 包含设施开设、柜容量档、OD—基地对分配及航线开通，\(F(x)\) 为固定费用。运营变量记为 \(v\)，包括完整路径流、骑手流、可用概率、空闲机队及提升变量。代码求解的是

\[
\min_{x\in X}\{F(x)+Q(x)\},\qquad
Q(x)=\min_v\{c^\top v+c_0:(x,v)\in\mathcal F\}.
\]

这里 \(\mathcal F\) 表示当前共同提升模型的全部线性和二阶锥约束，包括隐式定义的完整路径域。核查依据为 [模型说明](model.md)、[隐式 CG](implicit_cg.md)、`cpp/src/formulation.cpp::build_formulation` 与 `cpp/src/benders.cpp::solve_dual`。

- 这是一个含随机需求与占用近似的**确定性数学规划模型**。输入需求率、等待时间分布参数及柜容量阈值已进入确定的系数与约束。它没有场景集合、场景概率及按场景复制的运营决策，不是目前意义下的两阶段场景随机规划。
- 固定合法二元设计后，运营子问题是连续 SOCP。外层是整数设计 Benders，内层是连续 SOCP 的完整路径域列生成；不是整个整数模型的 branch-and-price。
- 对每个合法设计，全骑手配送给出后备解，故有相对完全追索。这里借用 recourse 一词描述参数化运营优化，不意味着存在随机规划的第二阶段场景。
- 共用柜容量、无人机与基地流量平衡形成运营耦合；不能未经重构就断言 \(Q(x)=\sum_rQ_r(x)\)，也不能将 OD 或基地直接当成独立随机场景。
- `aggregate` 的单任务/双任务类型是需求重构后的并存流量类别，不是互斥的随机情景。`prefix` 与 `aggregate` 需要分别保持原约束语义。
- 当前证书针对共同提升模型及声明的路径域成立，不自动证明其在线执行机制或随机排队近似精确。

一般凸子问题的分解背景是 Geoffrion（1972）[R2]；代码中的具体有效性还依赖以下公共矩阵与锥对偶条件，不能用一个方法名称代替证明。

## 2. 当前对偶割、BDD 与 L-shaped 的区别

### 2.1 普通锥对偶 Benders

代码在运营模型中保留设计变量的连续副本，通过固定行将其等于给定设计。其他矩阵、变量界、锥和路径列定义保持不变；设计只影响固定行右端。设 \(D\) 是这套共同模型的完整对偶可行域，则任意 \(d\in D\) 给出仿射函数

\[
Q(x)\ge \alpha(d)+\beta(d)^\top x.
\]

该不等式依赖弱对偶；不要求该对偶点在当前设计处最优。**割的全局有效性**与**割在当前设计处足够紧、能够认证子问题最优值**是两件事。后者需要强对偶/数值证书及原始—对偶间隙检查；完全追索本身不推出 Slater 条件，也不保证每个求解器返回的乘子可直接使用。

隐式列生成额外要求对完整路径域满足对偶平稳性。现有流程先求受限 SOCP，再完整标号定价，使用已证明的变量界修正数值残差及微负约化成本；未完成定价不能产生新的完整域证书。有限列 RMP 的最优值是完整运营问题的上界。不能因为 RMP 状态为 `Optimal` 就把其目标或对偶当成完整域下界。

### 2.2 Rahmaniani 等（2020）的 Benders Dual Decomposition

BDD 是一个有特定构造的算法，不是“求 Benders 子问题的对偶”的另一种说法。[R3] §2，作者提前在线版纸页3–4（本地 PDF 第4–5页），将主问题变量复制为 \(z\)，先加入 \(z=x\)，再将该等式按乘子松弛进入目标，同时对副本保留整数性。这样需要求解混合整数的拉格朗日内问题。

下面是对当前模型的独立有效性推导。2026-10-21 已实现 [独立 BDD 割质量诊断](bdd.md)；2026-10-22 增加完整域任意乘子 oracle 和[有限盒内 Kelley 乘子搜索](bdd_multiplier_research_20261022.md)。它们均未接入默认生产算法；当前搜索也不是原文近端 QP 或完整 BDD 的复现。令

\[
\mathcal F_I=\{(z,v)\in\mathcal F:z\in X\},\qquad
\ell(\lambda)=\inf_{(z,v)\in\mathcal F_I}
\{c^\top v+c_0-\lambda^\top z\}.
\]

对任何给定的 \(\lambda\) 及有效下界 \(\underline\ell\le\ell(\lambda)\)，均有

\[
\boxed{\theta\ge \underline\ell+\lambda^\top x.}
\]

证明：对任意合法 \(x\) 和其任意完整运营可行解 \(v\)，有 \(\underline\ell\le c^\top v+c_0-\lambda^\top x\)；对 \(v\) 取下确界即可。

这个证明不使用随机场景，也不依赖线性运营约束，因此可以作为后续锥版本研究的起点。但计算代价与认证条件显著不同：

1. 我们的内问题成为混合整数 SOCP，且设计和路径流一起变化；当前固定设计连续 CG 的结束证明不能直接认证它。
2. 需要完整域内问题的**全局下界**。内问题可行解目标是上界，不能放在上述截距中；受限列 MIP 的 best bound 也不保证下界于完整路径内问题。
3. 可以使用经证明的外松弛下界、完整模型 MISOCP 的求解器 best bound，或另行完整实现的 branch-and-price 证书。必须说明选择哪一种。
4. 用普通 Benders 乘子求一次整数内问题与进一步优化拉格朗日乘子，是 [R3] 区分的 strengthened Benders cut 和 Lagrangian cut 两个层次；不应混称。
5. [R3] §2 Theorem 1 说明在其 MILP 设置下保留整数副本可提升割；§2 Theorem 2 研究完整拉格朗日割族的根界；§4.3 Proposition 4 给出有已知误差上界时的保守截距修正。不能由此宣称少数、限时、受限列的锥内问题会在根节点求出我们问题的整数最优值。
6. [R3] 纸页4明确讨论非线性集合的可能扩展，但正文分析与实验主要围绕 MILP/LP 子问题。采用其思想前仍须给出本模型的映射、完整域界与停止条件。

后续实现以 [外部连续松弛 CG](continuous_root_cg.md) 直接认证共同模型的连续下界，再以独立完整域 BDD 探针检查整数结构增强。生产 CG 仍不预枚举完整路径；BDD 试验单独计入完整目录和模型构造时间。

### 2.3 L-shaped 与 partial Benders

Van Slyke–Wets（1969）[R4] 的原始 L-shaped 结构也出现在最优控制中，因此不能从名称断言它只适用于随机问题。当前需要避免混用的是**基于随机场景独立追索的具体机制和结论**。

Crainic 等（2021）[R5] 的 PBD 框架明确针对两阶段随机规划、连续追索，以场景保留、场景创造及其组合加强 L-shaped 主问题：§2 纸页416–417给出场景模型；§4.1 纸页419给出新主问题；§4.2 纸页420–421研究场景保留；§4.3 纸页421起研究人工场景。该文中的概率加权、场景支配和人工场景凸组合均需其场景模型条件。

对于当前模型：

- 不实施场景保留、场景创造、按 OD 套用场景权重、或场景独立 multi-cut。
- 部分投影/把部分运营变量留在主问题，是比该文更一般的分解选择；它并非逻辑上专属随机规划，但必须针对新的主从划分重新证明等价性、成本分摊和对偶割有效性。
- 当前用户要求保持精简主问题，因此不以“partial Benders”名义加回服务率 \(S\)、可用概率或库存变量。
- 外部资源 LP 求解后仅把有效线性不等式加入原 \(x,\theta\) 主问题，属于主问题有效不等式加强，不应包装成 [R5] 的场景 PBD。

## 3. 可优先采用的加强及其理论边界

| 方法 | 研究依据 | 本模型需要验证的条件 | 本轮采用原则 |
|---|---|---|---|
| 根节点先分离再整数求解 | McDaniel–Devine（1977）[R6]；[R1] §6.1 | 分数设计使用共同提升矩阵；完整域锥对偶、定价与残差检查成立 | 提高根阶段诊断与适应性，不能用分数运营解更新整数上界 |
| 非最优主问题候选生成割 | Côté–Laughton（1984）[R7]；Rei 等（2009）作者稿§2 [R8] | 候选可作为合法固定设计；子问题割独立认证；全局下界取 master best bound | 可减少早期主问题精度与时间，但保留全局搜索和最终收敛检查 |
| Magnanti–Wong 割选择 | Magnanti–Wong（1981）[R9]；[R1] §7 | 同一完整对偶域、正确锚点、核心点与辅助最优性；隐式对偶也要完整定价 | 先诊断失败原因，保留普通认证割作回退 |
| 独立核心点割 | Papadakos（2008）[R10] | 不依赖当前锚点仍必须完整域对偶可行；核心点条件和算法定义要明确 | 不将现有 MW 辅助求解误称为独立 Papadakos 实现 |
| 多候选点割池 | Rei 等（2009）[R8]；[R1] §6.3 | 每个候选分别完成有效子问题证书；选点允许启发式 | 可把邻域方案变成额外全局割，不将其邻域 bound 上报为全局界 |
| 资源 LP 对偶成本割 | [R1] §6.2 的有效不等式方法类别；本页§5的独立投影证明 | 任意完整运营可行解都可映射到乐观 LP；使用对偶可行上界而非未最优原始收益 | 不新增主问题运营变量，分别核查 K=1/K=2 与重复订单计数 |
| 全局树中分离 Benders 割 | [R1] §4；确定性设施选址研究可作为后续来源 | 合法全局/局部割范围、节点界、回调和超时语义 | 后续结构改造，不由本轮调度修改自动获得 |

### 3.1 非最优主问题的下界记账

设当前有效割主问题的最优值为 \(m_t^*\)，其求解器 best bound 为 \(b_t\)，候选目标为 \(m_t(x_t)\)。有

\[
b_t\le m_t^*\le z^*,
\]

但非最优候选 \(m_t(x_t)\) 不一定小于 \(z^*\)。因此应使用 \(L\leftarrow\max\{L,b_t\}\)，不能把候选目标当成新全局下界。目标截断存在时，须由已验证可行上界构造并保留所有潜在改进解。

早期放宽 master gap 的作用是更快取得割生成点，通常不直接增强同一主问题的数学松弛。若候选已经完成 CG、没有新违反割、但全局 \(U-L\) 仍大，应继续或收紧主问题求解，而不是按候选处的局部一致性宣布收敛。时间片、重试、累计耗时和精度收紧需要记录；有限时间预算下可以如实以 gap 退出，不能把启发式候选循环宣称为收敛证明。

### 3.2 连续松弛诊断应比较同一个模型

先比较完整路径模型连续松弛与根节点充分分离后的 Benders 松弛，而不是先猜测某一类割一定缺失。两侧需要一致的路径域、共同提升矩阵、有限变量界、航线大 M、目标单位和额外有效不等式。尤其不能将 Python 目录模型的不同紧界与 C++ 隐式模型直接当作“割数量差异”。

若两侧一致且已获得足够严格证书，根节点闭合后应重构相同的连续运营投影。仍存在显著差距时，先检查定价、预算、数值误差、未分离约束与模型不一致。若根界已接近相同完整模型连续界而整数 gap 仍大，同一连续对偶域内增加割不会在根节点越过该连续投影的极限；应研究整数有效不等式、更强的等价提升或 BDD 等整数感知加强。精确整数主问题搜索仍可能收敛，但这与根界加强是不同机制。

## 4. 当前核心点的可证明性与 Pareto 诊断

`cpp/src/benders.cpp::initial_core` 对原设计域 \(X\) 的构造可以给出一个简单概率证明，不必未经核查就把 Pareto 失败归因于“核心点随便取值”。此处概率只是证明凸组合的数学工具，不引入模型场景。

独立以概率 \(1/2\) 开设每个候选设施。每个开设的柜点均匀选择一个柜档。对每个 OD，均匀选择 \(L^2\) 个有序基地对及“不分配”共 \(L^2+1\) 个选项，若所选基地对的端点未全部开设，则置为不分配。每条允许航线在两端均开设时，再以概率 \(1/2\) 开通；禁止航线恒为零。

该分布只生成合法设计，并对每个 \(x\in X\) 给予正概率，其期望恰为：

\[
\bar x_i=\tfrac12,\qquad
\bar z_{is}=\frac{1}{2n_{levels}},\qquad
\bar a_{rlm}=\frac{\mathbf1_{l=m}/2+\mathbf1_{l\ne m}/4}{L^2+1},
\]

允许的同节点航线期望为 \(1/4\)，不同端点航线为 \(1/8\)，禁止航线为零，与代码相符。因此 \(\bar x\in\operatorname{relint}(\operatorname{conv}X)\)。固定为零的变量及柜档等式属于仿射包，不要求这些方向上有严格松弛。

当前更新 \(x^c\leftarrow0.50x^c+0.49x_t+0.01\bar x\) 在 \(x_t\in X\) 时保持相对内部性。若未来改变设计约束、增加预算/数目限制、删去设计域中的某些点，必须重验该证明。目标截断后的局部主问题域与原 \(X\) 要区分：相对原 \(X\) 进行割选择仍具有全域意义，不应无说明地改成另一个核心域。

因此 Pareto 诊断至少分别报告：

1. 普通 CG 是否完成，以及普通割锚点间隙。
2. 辅助最优面容差与辅助问题退出状态。
3. 辅助对偶残差、保守修正量、完整定价状态及最小遗漏约化成本。
4. 锚点误差、核心点处相对普通割的增益、修正造成的核心点损失。
5. 拒绝原因、预算/累计预算、辅助迭代及新增平稳性约束数量。

即使核心点有上述证明，现有辅助最优面使用数值带宽，割还经过保守下移；严格 Pareto 不受支配性质不应由 `accepted=true` 自动推出。可以准确称为“经过认证的 MW 风格加强割”。有效、锚点近似紧、核心点增益及严格 Pareto 是不同结论。

## 5. 外部加权资源 LP 的有效性

以下保留 [master_strengthening.md](master_strengthening.md) 已建立的完整输入域系数。令 \(g=(r,l,m)\)，\(j\) 为交付柜，\(\alpha_{gj}\ge0\) 为该类一次服务可归属节省的上界，\(\tau_{gj}\ge0\) 为其交付柜占用下界，\(C_j(z)\) 为所选柜档容量。考虑

\[
\begin{aligned}
B(a,z)=\max_{u\ge0}\quad&\sum_{g,j}\alpha_{gj}u_{gj}\\
\text{s.t.}\quad&\sum_j u_{gj}\le h_r a_g &&\forall g,\\
&\sum_g\tau_{gj}u_{gj}\le C_j(z)&&\forall j.
\end{aligned}
\]

这里 \(u\) 不进入原主问题。对每个完整运营可行解，将每次订单服务映射到相应 \(u_{gj}\)；任务中 AA 或重复柜点按订单出现次数累加。若逐路径成本分摊、总服务守恒及占用下界都已证明，则实际节省不超过 \(B(a,z)\)，因此 \(Q(x)\ge C_0-B(a,z)\)。这一步是模型特定证明，引用“网络流”或“Benders”不能代替它。

对应 LP 对偶为

\[
\begin{aligned}
\min_{\lambda,\mu\ge0}\quad&
\sum_g h_r a_g\lambda_g+\sum_j C_j(z)\mu_j\\
\text{s.t.}\quad&\lambda_g+\tau_{gj}\mu_j\ge\alpha_{gj}&&\forall(g,j).
\end{aligned}
\]

对**任何经过可行性认证的对偶点**，有

\[
\boxed{\theta\ge C_0-
\sum_g h_r a_g\lambda_g-
\sum_j C_j(z)\mu_j.}
\]

对偶约束独立于设计，因此固定系数后是全局有效行；对偶不必最优，最优性影响强度而非弱对偶有效性。相反，收益最大化 LP 的一个未证明最优的原始可行目标仅是收益下界，直接用 \(C_0-\text{该目标}\) 会错误地抬高运营成本下界。

零占用系数不需要除法：例如 \(\lambda_g\ge\max_j\alpha_{gj}\)、\(\mu=0\) 始终给出安全对偶点；也不能为了避免除零而删除允许的服务。对于任意有限非负候选 \(\mu\)，逐组重设 \(\lambda_g=\max\{0,\max_j(\alpha_{gj}-\tau_{gj}\mu_j)\}\)，并保守向上修正舍入误差，可恢复对偶可行性。因此辅助 LP 限时返回的候选仍有可能被修复为有效割，但未修复或未验证的返回值不能直接使用。数值修正应使对偶约束更可行，使被减去的收益上界不降低。

必须同时验收整数与分数设计、K=1/K=2、AA、重复柜、跨基地、禁用航线、零时间参数及不同生成列池；所有系数从完整输入路径域的必要条件得出，不取当前列池极值。分数设计的有效性要对共同提升模型的 \(u\) 映射验证，不由二元设计证明自动推定。

## 6. 邻域割池与实验归因

邻域搜索本身缩小战略域，受限列再缩小运营域，两者的内部 best bound 均不能直接作为原问题全局下界。Rei 等（2009）[R8] 给出的可借鉴机制是用若干不同战略点生成额外全局割。我们因完全追索，不直接采用其不可行邻域排除路线。

可采用的扩展为：低预算受限列搜索生成多个不同设计；去重并选择候选；对选中设计运行完整域固定设计 CG；仅将已认证对偶割加入全局主问题。候选不必改善 incumbent 才有割价值，候选来源的局部限制不应写进全局主问题。该流程仍保留全局 master 搜索，不能只循环邻域而宣称全局收敛。

实验应分开比较 master 调度、根分离、资源 LP、Pareto 诊断/修复、邻域割池，至少保留基线与单因素开关。报告相同时间预算下的有效 LB、已验证 UB、gap，以及完整 CG 数、普通/辅助/资源割数量、未完成定价次数、各阶段耗时。多项同时更改的一次结果不能独立证明其中某种方法有效。

## 7. 文献与核查记录

本轮是针对已识别文献的定向核查，不是声称穷尽全部 Benders 变种的系统综述。对来源的使用边界如下。

- **[R1]** Rahmaniani, R., Crainic, T. G., Gendreau, M., Rei, W. (2017). *The Benders decomposition algorithm: A literature review*. EJOR 259(3), 801–817. DOI [10.1016/j.ejor.2016.12.005](https://doi.org/10.1016/j.ejor.2016.12.005)。已读本地全文；重点 §5 纸页807、§6.1 808、§6.2–6.3 809、§7 810–811。分类依据来自综述，具体有效性依赖原文及本模型推导。
- **[R2]** Geoffrion, A. M. (1972). *Generalized Benders decomposition*. JOTA 10, 237–260. DOI [10.1007/BF00934810](https://link.springer.com/article/10.1007/BF00934810)。出版社摘要及 [作者全文](https://www.anderson.ucla.edu/faculty_pages/art.geoffrion/home/docs/GBD.pdf) 首部核查，用作一般凸子问题分解背景；未将原文全部正则条件视为本项目已自动满足。
- **[R3]** Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., Rei, W. (2020). *The Benders Dual Decomposition Method*. Operations Research 68(3), 878–895. DOI [10.1287/opre.2019.1892](https://pubsonline.informs.org/doi/10.1287/opre.2019.1892)。已读本地提前在线版全文重点 §2、§4.1–4.3；本文页码采用该版纸页1–18，本地 PDF 有一页下载封面，故 PDF 页=纸页+1。
- **[R4]** Van Slyke, R. M., Wets, R. (1969). *L-Shaped Linear Programs with Applications to Optimal Control and Stochastic Programming*. SIAM J. Applied Mathematics 17(4), 638–663. DOI [10.1137/0117061](https://epubs.siam.org/doi/10.1137/0117061)。出版社摘要与 [作者保存全文](https://www.math.ucdavis.edu/~rjbw/mypage/Stochastic_Optimization_files/VnsW69_Lshape.pdf) 首部核查，用于术语及适用结构边界。
- **[R5]** Crainic, T. G., Hewitt, M., Maggioni, F., Rei, W. (2021). *Partial Benders Decomposition: General Methodology and Application to Stochastic Network Design*. Transportation Science 55(2), 414–435. DOI [10.1287/trsc.2020.1022](https://pubsonline.informs.org/doi/10.1287/trsc.2020.1022)。已读本地全文重点 §2、§3、§4.1–4.3。正式卷期为2021，在线发表为2020-12-24。
- **[R6]** McDaniel, D., Devine, M. (1977). *A Modified Benders' Partitioning Algorithm for Mixed Integer Programming*. Management Science 24(3), 312–319. DOI [10.1287/mnsc.24.3.312](https://pubsonline.informs.org/doi/10.1287/mnsc.24.3.312)。出版社元数据/摘要核查，根先行分离细节由 [R1] 和 [R8] 交叉核对；不将本文调度参数归于原作者。
- **[R7]** Côté, G., Laughton, M. A. (1984). *Large-scale mixed integer programming: Benders-type heuristics*. EJOR 16(3), 327–333. DOI [10.1016/0377-2217(84)90287-X](https://www.sciencedirect.com/science/article/pii/037722178490287X)。出版社摘要核查；非最优主问题候选的说明由 [R8] 作者稿§2交叉支持，本文下界记账式为独立弱对偶推导。
- **[R8]** Rei, W., Cordeau, J.-F., Gendreau, M., Soriano, P. (2009). *Accelerating Benders Decomposition by Local Branching*. INFORMS Journal on Computing 21(2), 333–345. DOI [10.1287/ijoc.1080.0296](https://pubsonline.informs.org/doi/10.1287/ijoc.1080.0296)。已核查 [作者稿](https://users.encs.concordia.ca/~chvatal/gendreau.pdf) §1–3，作者稿纸页3–7；作者稿页码不可当成期刊页码。
- **[R9]** Magnanti, T. L., Wong, R. T. (1981). *Accelerating Benders Decomposition: Algorithmic Enhancement and Model Selection Criteria*. Operations Research 29(3), 464–484. DOI [10.1287/opre.29.3.464](https://pubsonline.informs.org/doi/10.1287/opre.29.3.464)。出版社摘要及 [R1] §7 核查。本文核心点概率构造与数值证书边界为针对当前代码的独立推导。
- **[R10]** Papadakos, N. (2008). *Practical enhancements to the Magnanti–Wong method*. Operations Research Letters 36(4), 444–449. DOI [10.1016/j.orl.2008.01.005](https://www.sciencedirect.com/science/article/abs/pii/S0167637708000102)。出版社摘要/引言可见文本核查；单独访问非摘要页面一度返回403，不声称本轮取得完整PDF。

检索日期均为2026-10-21。使用 paper-lookup 技能的定向 DOI 核查流程：对 Crossref `GET /works/10.1287%2Fopre.2019.1892` 的请求因配置代理 `127.0.0.1:7897` 连接失败而没有取得元数据；随后使用出版社页面和已有本地PDF。这个接口失败不表示论文不存在，也不影响上述三份本地全文读取。网络检索主要按精确标题/DOI定位作者与出版社来源，未将第三方索引摘要作为方法定理证据。

本地文献（未修改）的复现信息：

| 文献 | 本地文件 | SHA-256 |
|---|---|---|
| R1 | `/Users/songtaowang/Zotero/storage/EPLLLWXT/Rahmaniani et al. - 2017 - The Benders decomposition algorithm A literature review.pdf` | `40cce53b3ebd2613b65d288b4df14a9f114e5a98887726f33a0e60db8a640248` |
| R3 | `/Users/songtaowang/Zotero/storage/TRYDHFT7/Rahmaniani et al. - 2020 - The Benders Dual Decomposition Method.pdf` | `2f5307538562d9e60c452b6bf05a4d4f040b8c7b3d3a43e5e39134fb50f30b96` |
| R5 | `/Users/songtaowang/Zotero/storage/MN2KAHZY/Crainic et al. - 2021 - Partial Benders Decomposition General Methodology and Application to Stochastic Network Design.pdf` | `61757801b660368483715269725c9b6c40a175bafdade234d9615def87331fd0` |

使用本机 `pdftotext -layout` 提取全文到临时目录并按换页符定位页码。以上证据支持方法选择和条件核验，不构成当前实现已提速的实验证明。
