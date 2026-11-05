# L2 强割构造：一手文献与可迁移边界

核对日期：2026-10-28（Asia/Shanghai）。范围：固定任务率、origin 分配，2 基地、20 柜点、50 OD；固定战略设计的运营子问题为 SOCP，普通对偶由原始求解器读取。本稿仅给理论构造和分离预算建议，没有修改算法、启动求解器或安排消融/固定主问题测试。

优先方向是**对资源定价后的排队值函数使用扩展多拟阵不等式（EPI）**。这是有条件的构造：先有覆盖完整路径域的资源价格下界，再对其中保留的分配整数性凸化。文献支撑 EPI 的有效性与分离方法；本文并不据此宣称当前 MLND 已经具备所需资源价格证书，或新割必然提升最终全局 LB。

## 1. 五篇最相关原文

### [1] Atamtürk–Narayanan (2008)：一般集合函数的 EPI 与 greedy 分离

**Polymatroids and mean-risk minimization in discrete optimization**, *Operations Research Letters* 36(5), 618–622. DOI: 10.1016/j.orl.2008.04.006。[作者网站正式全文](https://atamturk.ieor.berkeley.edu/pubs/_published/orl36-2-2008.pdf)。

精读位置：第 2 节，期刊 p.619，Proposition 1、其后的 Edmonds 极点/分离描述及 Remark 1；第 3 节 Theorem 1。第 2 节从一般实值集合函数出发，允许先减去空集值；EPI 系数可以为负，因而不要求本项目的排队成本下界单调非负。前缀增量给出极端 EPI，按待分离点坐标降序生成最违反的 EPI。这是本稿主线最直接的来源。

### [2] Atamtürk–Gómez (2020)：二元结构与锥优化的连接

**Submodularity in Conic Quadratic Mixed 0–1 Optimization**, *Operations Research* 68(2), 609–630. DOI: 10.1287/opre.2019.1888。[作者预印本 v3](https://arxiv.org/pdf/1705.05918v3)，[出版社版本信息](https://pubsonline.informs.org/doi/10.1287/opre.2019.1888)。

精读位置：预印本 §3，Propositions 1–2（PDF p.9），明确给出二元子模 epigraph 的凸包描述与 O(n log n) 分离；§2.5 有排队选址背景；§4–6 讨论混合连续变量及提升。可迁移的是“抽取二元子模部分，再生成有效不等式”的机制。其平方根/旋转锥公式不能直接当成下文负共轭值函数的公式；也不能从单一结构集合的凸包推广为整个 MLND 可行域的凸包。

### [3] Codato–Fischetti (2006)：部分设计冲突与缩减条件

**Combinatorial Benders' Cuts for Mixed-Integer Linear Programming**, *Operations Research* 54(4), 756–766. DOI: 10.1287/opre.1060.0286。[出版社原文页面](https://pubsonline.informs.org/doi/10.1287/opre.1060.0286)，[Fischetti 上传的原文全文](https://www.researchgate.net/publication/220244181_Combinatorial_Benders'_Cuts_for_Mixed-Integer_Linear_Programming)。

精读位置：作者稿 §1 的式 (7) 与 MIS 缩减，§2 的部分二元赋值，§4 的实现。贡献是从足以证明不可行的一小组条件生成组合割；整数点可用删除条件的方式缩小冲突。原文以 LP 条件系统为核心。SOCP 中应使用完整域的可验证界/不可行证书，不能把数值失败、限时无解或非紧对偶当成逻辑冲突。

### [4] Rahmaniani et al. (2020)：少量整数性的 Lagrangian 割

**The Benders Dual Decomposition Method**, *Operations Research* 68(3), 878–895. 作者为 R. Rahmaniani、S. Ahmed、T. G. Crainic、M. Gendreau、W. Rei。DOI: 10.1287/opre.2019.1892。[作者机构原文 CIRRELT-2019-27](https://www.cirrelt.ca/documentstravail/cirrelt-2019-27.pdf)，[出版社版本信息](https://pubsonline.informs.org/doi/10.1287/opre.2019.1892)。

精读位置：§2.1、Proposition 1、Theorem 1，PDF pp.7–8。复制战略变量并松弛其固定等式后，可仅保留一个变量子集的整数性。原文的增强量来自内层整数与连续问题的最优值差。原文是 MILP；下文 SOCP 迁移只使用弱对偶，不借用其 MILP 性能或有限收敛结论。

### [5] Sherali–Lunday (2013)：原始 RHS 扰动选割

**On generating maximal nondominated Benders cuts**, *Annals of Operations Research* 210, 57–72；2011 年在线发表。DOI: 10.1007/s10479-011-0883-6。[出版社原文页面](https://link.springer.com/article/10.1007/s10479-011-0883-6)，[Lunday 上传的原文全文](https://www.researchgate.net/publication/241028200_On_generating_maximal_nondominated_Benders_cuts)。

精读位置：§2 的式 (11)–(20)、Theorem 1、Corollary 1 及 Algorithm 4a/4b。加权次目标可以落实为原始子问题 RHS 扰动，避免单独构造普通对偶优化。原文的精确优先级及 maximal-cut 结论有其 LP/权重条件；有限扰动的 SOCP 乘子不能自动称为 Pareto 最优。

## 2. 主线：排队值函数的 EPI

以下是针对候选值函数的数学迁移，不是上述论文已经证明了 MLND 的声明。

对某一固定资源价格向量，假定已证明

\[
Q(x)\ge a+\beta^\top x+\sum_{l=1}^{2} f_l(S_l(\alpha)),
\qquad
f_l(S)=-H_{F_l}^*\!\left(b_l+\sum_{r\in S}w_{rl}\right),
\tag{A}
\]

其中 \(F_l\ge0,w_{rl}\ge0\)，且

\[
H_F^*(s)=
\begin{cases}
0,&s\le F,\\
(\sqrt{s}-\sqrt F)^2,&s>F.
\end{cases}
\]

若原权重 \(B_{rl}<0\)，用 \(u_{rl}=1-\alpha_{rl}\)；否则 \(u_{rl}=\alpha_{rl}\)。于是

\[
\sum_r B_{rl}\alpha_{rl}
=\sum_{B_{rl}<0}B_{rl}+\sum_r|B_{rl}|u_{rl}.
\]

常数必须进入 \(b_l\)，且每个基地允许有不同的互补映射。不能只把负权重取绝对值而丢掉常数。

**子模性。** \(-H_F^*\) 是整个实轴上的凹函数：在 \(s>F>0\) 时导数为 \(-1+\sqrt{F/s}\)，二阶导数非正；在阈值处与常数段衔接。\(F=0\) 时为 \(-\max(s,0)\)，同样凹。非负模权重上的凹函数具有递减边际，所以 \(f_l\) 子模。这一步与 [1] 用凹性证明子模性的机制一致，函数本身由本项目另外给出。

任一排列 \(\sigma\) 定义 \(S_i=\{\sigma(1),\ldots,\sigma(i)\}\) 和

\[
\pi_{\sigma(i)}=f_l(S_i)-f_l(S_{i-1}).
\]

由 [1] 的一般 EPI 结论，

\[
t_l\ge f_l(\varnothing)+\sum_i\pi_{\sigma(i)}u_{\sigma(i),l}
\tag{EPI}
\]

对二元 epigraph 有效。若不增加 \(t_l\)，直接将两个基地的一条 EPI 代入 (A)，得到主问题中一条关于原设计与 \(\theta\) 的线性割。负边际系数合法；空集常数不能省略。

**精确分离与凸包范围。** 对当前 \(\bar u\)，按 \(\bar u\) 降序排序即获得该基地最违反的 EPI。[1] §2 给出一般 greedy 分离，[2] Proposition 2 给出对应复杂度。全部排列给出二元函数 epigraph 的凸包：按该排列，\(\bar u\) 可分解为嵌套前缀指示向量的凸组合，EPI 的值恰为这组顶点函数值的同权平均。这也说明结论不依赖函数单调性。加入跨基地互斥、设施和容量约束以后，EPI 仍有效，但这组不等式不必刻画交集的全部凸包。

**必要边界。** (A) 必须先是完整路径域、原始订单出现次数及基地流量平衡下的全局下界。EPI 不能补救错误的资源定价或被省略的第二单；这里凸化的是一个已认证下界函数，不是宣称得到了 \(Q\) 的精确凸包。其有效域是原整数设计及其 epigraph 的凸包，可能不保留旧连续松弛的所有点。只有原主问题重新求解所得认证界才可记为 LB 提升。

## 3. 三个备选机制及其限制

### 部分设计的条件成本割

令 \(D_S(x)\) 为二元坐标与某模式在集合 \(S\) 上的失配数。若完整域放松问题已经证明
\(x_S=\bar x_S\Rightarrow Q(x)\ge L_S\)，并有全局 \(Q(x)\ge L_0\)，取 \(L_S\ge L_0\) 可得

\[
\theta\ge L_S-(L_S-L_0)D_S(x).
\tag{C}
\]

这将 [3] 的部分条件推断迁移为成本割：匹配时使用 \(L_S\)，不匹配时由 \(L_0\) 保证。删除一个条件后必须重新证明同一个成本门槛；固定完整设计的 \(Q(\bar x)\) 不能直接作为部分模式的 \(L_S\)。改进方向是减少条件或认证更小的失配罚项；不能无证据降低 big-M。若采用 incumbent cutoff 排除模式，必须另标其“保留改进解”的范围，不能当成对全部原始可行解有效的成本割。

### 少量整数性的 Lagrangian 截距

对完整域联合放松 \(\mathcal R\)，只保留坐标集合 \(J\) 为整数，定义

\[
\phi_J(\beta)=\inf\{c^\top y-\beta^\top z:(z,y)\in\mathcal R, z_J\in\mathbb Z^J\}.
\]

任何认证 \(\underline\phi_J\le\phi_J\) 给出
\(\theta\ge\underline\phi_J+\beta^\top x\)。这是 [4] Proposition 1 的弱对偶迁移；SOCP 原始约束不改变该不等式方向。受限路径内层的 best bound 和内层可行目标均不能充当该截距。一个小型完整域 MISOCP 的全局 best bound可以。

若 \(\beta\) 是整数锚点的紧连续 recourse 支持斜率，且该锚点属于内层域，它本身会封住同斜率的截距提升；因此不能把“保留少量整数性”自动等同于增强。分数锚点、不同斜率或真正存在离散间隙才可能产生收益；[4] §2.1 也明确讨论整数与分数锚点的区别。

### RHS 扰动与多点选割

保持原 SOCP 目标、锥矩阵和所有路径列系数不变，只移动可行 RHS/设计锚点，从该原始问题读取乘子，再针对原模型完整定价及保守修正，可获得有效割。这不需要额外普通对偶优化，但有限扰动不保证在原锚点紧，也不保证 Pareto 性。[5] 的 LP 最大非支配结论不能直接移植。

特别地，双目标 \(r(x)^\top\pi+\mu r(c)^\top\pi\) 与在 \(x_\epsilon=(x+\mu c)/(1+\mu)\) 处选原生乘子等价（当 \(r\) 为仿射、对偶可行域不随设计变化）。因此它与当前内部点机制属于同一选割思路，单纯换 \(\mu\) 不是新的整数性增强。已有多条完整域证书可以在当前点及少量代表设计上直接计算割值，筛出互补的割；此筛选只评价已认证仿射函数，不产生新的全局最优值证明。

## 4. L2 的分离规模与建议预算

以下均为构造预算建议，不是实测时间，也不是新实验任务。

| 机制 | L2 结构规模 | 建议限制与作用 |
|---|---|---|
| 排队 EPI | 每基地最多 50 个二元头单分配坐标；两次排序、各 51 个闭式函数值 | 每个已认证价格向量先分离一条两基地合并割；保留少量价格向量。每向量 O(2×50 log 50)，无需新增优化求解；完整域价格证书的成本另计。 |
| 部分模式成本割 | 先考虑 2–4 个有明确资源耦合的分配文字，其他战略量自由/放松 | 只在已有闭式资源界或可靠完整域凸放松能给 \(L_S\) 时采用；不要枚举所有 OD 子集。删条件的辅助求解若未认证就停止缩减。 |
| 小整数 Lagrangian | \(|J|=1–2\) 时至多 2–4 个二元分支；\(|J|=3\) 已有 8 个 | 若需原 SOCP 分支求解，每次总上限 5 秒、整轮上限 30 秒可作预算起点。只有所有分支有有效LB时才能取其最小值；未解决分支不得直接丢弃。 |
| 原始 RHS/多点选割 | 一个额外原始 SOCP 或对已有证书做少量点积 | 沿用现有每次 5 秒、总预算 5% 的上限即可；其定位是改善证书选择，不承诺跨越连续松弛界。 |

建议先把 (A) 的资源价格证书、符号互补常数、两基地相加时的运营成本计数写清，再采用 EPI；其分离成本最小且保留了 50 个分配变量的整数信息。部分模式与小整数 Lagrangian 是预算更高的备选，RHS 扰动主要补充斜率。所有方向都保留原任务模型和 native-only 普通对偶要求。

## 5. 本稿的来源和完成范围

已浏览核对上列五篇原文/作者版本的相应章节；没有使用综述来替代原文。预印本页码与正式期刊页码已区分。当前实现边界依据本地 [统一模型](unified_task_algorithm_20261027.md) 与 [已实现界改进](bounds_improvement_20261028.md)。本文的 (A)、(C)、SOCP Lagrangian 迁移和预算属于针对本项目的推导或建议，不是文献已经在 MLND 上验证的结果。
