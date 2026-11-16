# E1 队列次模割：可直接引用的理论文献

核对日期：2026-11-12。范围：针对已构造的队列成本下界，核对“凹函数复合非负模函数—次模性—扩展多面体不等式—排序精确分离”的引用链。仅使用原论文、作者公开稿和出版社书目信息。应用论文由另一份研究笔记单独整理；本稿不作穷尽性综述。

## 1. 最直接的近期引用

**Yu, Qimeng, and Simge Küçükyavuz (2023). Strong valid inequalities for a class of concave submodular minimization problems under cardinality constraints. Mathematical Programming 201:803–861. DOI: [10.1007/s10107-022-01921-5](https://doi.org/10.1007/s10107-022-01921-5).** 出版社确认正式年份为 2023；公开作者稿为 2022-12-20 修订版。以下页码均指该 40 页作者稿，不冒充期刊排版页码。[出版社](https://link.springer.com/article/10.1007/s10107-022-01921-5)，[作者全文](https://optimization-online.org/wp-content/uploads/2021/03/draft.pdf)。

| 位置 | 核实内容 | 对 E1 的用途 |
|---|---|---|
| §1，p.1；§2.1 Lemma 2.1，p.3 | 研究凹函数复合非负加权和；Lemma 给出凹函数的增量随起点递减，不要求凹函数递增 | 支持成本函数可递减、边际可为负的次模论证 |
| §2.2，pp.3–4，式 (3) | 回顾一般次模集合函数的前缀边际 EPI | 对应本研究的队列 epigraph 割 |
| §3，始于 p.4；Proposition 3.6，p.8 | 在基数约束、两种权重的结构下精确提升 EPI | 作为“利用额外离散结构进一步加强”的近期理论参照 |
| §7，pp.34–37 | 将所提不等式用于均值—风险问题的 branch-and-cut，并比较算法 | 可参考理论结果与算法验证的组织方式 |

**适用边界：**本研究非负版本可令其一元凹函数为 `g(s)=-Phi_F(s)`，权重为固定任务节约系数。论文的特殊 lifting 结果带有基数和权重条件，不能因为形式相似就直接宣称覆盖本研究的多基地互斥约束；资源价格产生负系数后的补变量和缺失支持门控，仍需本研究自己的证明。[作者全文 §2–3、§6](https://optimization-online.org/wp-content/uploads/2021/03/draft.pdf)。

## 2. 有效性与“为什么排序最强”的直接出处

**Atamtürk, Alper, and Vishnu Narayanan (2022). Submodular function minimization and polarity. Mathematical Programming 196:57–67. DOI: [10.1007/s10107-020-01607-w](https://doi.org/10.1007/s10107-020-01607-w).** 在线发表日期为 2021-01-28，正式卷期为 2022，按卷期引用宜写 2022。[出版社](https://link.springer.com/article/10.1007/s10107-020-01607-w)。

核对版本：[作者公开全文，arXiv:1912.13238，11 页，文内版本日期 December 2019; November 2020](https://arxiv.org/pdf/1912.13238)。以下为该稿页码：

- §2，Proposition 2，p.3：`pi*x <= z` 对二元 epigraph 的凸包有效，当且仅当 `pi` 属于对应集合函数多面体。
- §2，Proposition 3，p.3：极点与齐次非平凡面之间的关系。
- §2，Proposition 4，pp.4–5：次模情形下非平凡面的齐次性质，从而得到与 Lovász 延拓一致的凸包描述。
- **§2，Remark 2，p.5，式 (6)**：分离等价于在扩展多面体上最大化 `pi*xbar`；将 `xbar` 降序排列后，取各前缀的函数差值即得到最违反 EPI。此处没有名为 “Algorithm 1” 的算法，不应使用该编号。

**适用边界：**这个结论证明的是固定集合函数、固定分数点的 EPI 精确分离。它不能自动证明完整运营值函数 `Q` 次模，也不能证明排序同时优化资源价格、跨基地联合凸包或全部可能的有效割。[作者全文 §2，特别是 Remark 2](https://arxiv.org/pdf/1912.13238)。

## 3. 保留一篇基础来源

**Atamtürk, Alper, and Vishnu Narayanan (2008). Polymatroids and mean-risk minimization in discrete optimization. Operations Research Letters 36:618–622. DOI: [10.1016/j.orl.2008.04.006](https://doi.org/10.1016/j.orl.2008.04.006).** [作者网站上的期刊全文](https://atamturk.ieor.berkeley.edu/pubs/_published/orl36-2-2008.pdf)。

定位：§2，期刊 p.619（PDF 第 2 页），Proposition 1 给出线性 epigraph 不等式与扩展多面体的对应关系；紧随其后的段落给出由排列前缀差值描述极点的 Edmonds 结论。这是基础引用。对“不要求单调增加的一般次模 epigraph”及“最违反割如何排序”，建议同时引用上面的 2022 年论文，而不要把 2008 年文中特定均值—风险函数的全部假设照搬到本研究。[期刊全文 §2](https://atamturk.ieor.berkeley.edu/pubs/_published/orl36-2-2008.pdf)。

## 4. 在本研究中如何使用这些引用

下述对应关系是本研究的推导，而非文献对 MLND 模型已证明的结论。对固定基地，若

\[
f_l(A)=-\Phi_F\!\left(\sum_{q\in A}S_{ql}\right),\qquad S_{ql}\ge0,
\]

则先由本研究证明 `Phi_F` 凸，从而 `-Phi_F` 凹；其与非负模函数的复合具有次模性。该结构可用 Yu–Küçükyavuz 的讨论佐证。接着，任意排列前缀差值构成有效 EPI，以及按照当前分数变量降序排列能实现该 EPI 族的精确分离，可引用 Atamtürk–Narayanan (2022) §2。[Yu–Küçükyavuz 全文](https://optimization-online.org/wp-content/uploads/2021/03/draft.pdf)，[Atamtürk–Narayanan 全文](https://arxiv.org/pdf/1912.13238)。

论文仍需自行建立“原运营问题的成本不小于此队列集合函数下界”这一连接；不能仅证明次模性就跳过原问题到下界的有效性推导。带符号价格方案必须先说明补变量变换与门控后的新集合函数，再应用一般 EPI 定理。

## 5. 容易混淆的两种文献路线

本研究使用 **次模成本最小化的 epigraph 下界**，形式为 `z >= affine(x)`。覆盖收益、设施捕获收益等文献常研究 **次模收益最大化的 hypograph 上界**，形式为 `eta <= affine(x)`；后者常使用 Nemhauser–Wolsey 型 submodular inequalities。两类文章都可能称其为 “submodular cuts”，却不能互换公式或分离定理。Atamtürk–Narayanan (2022) 在引言及 §2–3 明确区分最小化、最大化和一般集合函数近似；本研究的排序最强结论应落在其 §2 的 EPI 分离结果上。[作者全文](https://arxiv.org/pdf/1912.13238)。

建议正文的主引用组合为 **Yu–Küçükyavuz (2023) + Atamtürk–Narayanan (2022)**；2008 年论文用于交代基础来源。对本研究已有 cut 的贡献描述应落在特定队列下界的构造、有效连接和价格/支持处理上，不能将通用 EPI 或 Edmonds 排序分离宣称为新方法。
