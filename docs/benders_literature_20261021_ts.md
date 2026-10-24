# Transportation Science 近年选址/Benders 文献核对

核对日期：2026-10-21。范围为本地 Zotero、2023–2026 年 Transportation Science 的 Crossref 元数据及出版社/作者公开原文；随后补充内容高度相符但期刊或年份不同的论文。本轮不改数学模型，不以摘要推断未读到的算法细节。

## 1. 定位结果：尚不能唯一确认用户记忆中的论文

| 候选 | 已核对的书目信息 | 与描述的匹配程度及原文覆盖 |
|---|---|---|
| Rahmati, Üster, *Time-Phased Relief Supply Network Planning for Foreseen Disasters* | Transportation Science 60(5):784–805，2026；在线 2026-06-10，DOI [10.1287/trsc.2025.0217](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0217) | 补检发现的强候选：配送中心选址、时段网络、Benders、主问题解修正和多种 surrogate constraints 都与线索吻合。已核对出版社摘要、作者学校主页及 GERAD 讲座；未取得全文，尚不能核对割的公式和证明。 |
| Li, Taherkhani, Hewitt, Alumur, *Demand-Driven Hub Network Design Under Uncertainty for Less-Than-Truckload Carriers* | Transportation Science，2026-04-03 在线，DOI [10.1287/trsc.2025.0180](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0180) | 年份、期刊、hub、增强 Benders 均匹配；目前只核对出版社摘要，未取得全文，不能据此宣称已经核对具体割公式。 |
| Silva, Carvalho, Jena, *Dynamic Facility Location Under Cumulative Customer Demand* | Transportation Science 60(1):46–63，2026，DOI [10.1287/trsc.2025.0104](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0104) | 年份、期刊、facility 匹配；已读作者 2024 预印本 §5.1，主线是解析求解子问题，未证实属于“多种加强/稳定化策略”的那篇。 |
| Calamita, Ljubić, Palagi, *Benders decomposition for congested partial set covering location with uncertain demand* | **European Journal of Operational Research** 330(1):26–46，2026，DOI [10.1016/j.ejor.2025.08.041](https://doi.org/10.1016/j.ejor.2025.08.041) | 本地现有论文中，内容最匹配“拥堵选址＋Benders 加强＋实现策略”，但**不是 Transportation Science**；已读正式版 §3–5。 |
| Wu, Chen, Cordeau, Jans, *Machine Learning-Empowered Benders Decomposition for Flow Hub Location in E-Commerce* | **INFORMS Journal on Computing**，2025，DOI [10.1287/ijoc.2023.0367](https://doi.org/10.1287/ijoc.2023.0367) | 本地有稿件，hub 与多种加速相符，但期刊不同；摘要涉及聚类重构、变量消除、学习，尚不足以支持具体实现。 |
| Bayram, Yıldız, Farham, *Hub Network Design Problem with Capacity, Congestion, and Stochastic Demand Considerations* | Transportation Science 57(5):1276–1295，2023，DOI [10.1287/trsc.2022.0112](https://doi.org/10.1287/trsc.2022.0112) | 用户此前已附的论文；期刊/问题最贴近，年份较早。已核对 §4.1–4.3，尤其最大流最小割成立的前提。 |
| Taherkhani, Alumur, Hosseini, *Benders Decomposition for the Profit Maximizing Capacitated Hub Location Problem with Multiple Demand Classes* | Transportation Science 54(6):1446–1470，2020，DOI [10.1287/trsc.2020.1003](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2020.1003) | 年份不符，但“hub＋多种加速”非常相符；摘要明确两阶段子问题算法、两种强化分离、变量固定，并另行扩展随机模型。不能把随机扩展中的 SAA 加速等同于确定性主算法。 |

另外核对了 Malladi–Muthuraman (2024) *Facility Location Problem: Modeling Joint Disruptions Using Subordination*，DOI [10.1287/trsc.2023.0103](https://pubsonline.informs.org/doi/10.1287/trsc.2023.0103)。其摘要重点为联合中断的随机过程建模和校准，未发现足够证据将其识别为所找的 Benders 加速论文，故不作为实施依据。

检索是有边界的候选核对，不代表穷尽 TS 全部论文。Crossref 的 journal/ISSN=0041-1655、2023-01-01 至 2026-12-31、query=Benders/location 检索分别返回 4/3 条元数据匹配；全文不含标题关键词的论文仍可能漏检。

补检 Rahmati–Üster 的一手来源为[出版社](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0217)、[作者学校主页](https://s2.smu.edu/~uster/)和[GERAD 讲座](https://www.gerad.ca/en/events/2468)。出版社 PDF 链接返回摘要页，本机 Zotero 未找到该文；未绕过访问限制，也未依据摘要实施某个具体割。摘要明确覆盖最坏需求，不足以据此认定其子问题具有本项目的完全追索性。它进一步说明第二篇仍不能唯一识别；若用户认出题名，可沿该候选继续核对全文。

## 2. 最贴近当前瓶颈的正式原文：Calamita et al. (2026)

正式版本地路径：`/Users/songtaowang/Zotero/storage/4GNDM8BF/Calamita et al. - 2026 - Benders decomposition for congested partial set covering location with uncertain demand.pdf`。下述页码均为期刊页码；对应 PDF 第 5–8 页。

### 2.1 它的主问题不是只保留选址变量

§3.1，p.30，式(7)：主问题保留开设变量 `y_i`、设施负荷 `v_i` 和拥堵成本上图变量 `u_i`，保留完整目标及 perspective 旋转二阶锥约束 `v_i² ≤ u_i y_i`。它投影掉客户相关的分配和鲁棒对应变量。因此，主问题本身已经包含重要的负荷—拥堵关系，子问题则是 LP。

这与当前 MLND 的分工不同：当前主问题为战略变量与运营成本代理 `theta`；拥堵、路径流和实际服务均在 SOCP 子问题里。论文的性能结果不能被解释为“仅有选址变量和一个目标值下界也足够”。但也不能据此未经授权把 `S` 或所有负荷变量重新加回当前主问题。

### 2.2 它的割是覆盖可行性割，不能直接照搬

§3.2，p.30：原问题必须达到覆盖目标 `D`，固定主问题点后可能无法完成。作者改解一个始终可行且有界的最大覆盖 LP，记值为 `phi'(y,v)`；零分配保证该辅助 LP 可行。如果最大覆盖仍小于 `D`，就应排除主问题点。

§4，p.31，式(10)：最大化 LP 值函数的对偶仿射上界产生

`phi'(y_bar,v_bar) + r_y·(y-y_bar) + r_v·(v-v_bar) >= D`。

其机制是“覆盖量上界必须达到强制覆盖需求”。当前 MLND 有骑手兜底，固定合法战略点后的运营子问题总可行；不能把服务不足当作不可行并删掉该点。若借鉴覆盖/最大流机制，必须重新证明无人机**节约成本的上界** `saving_upper(x)`，进而添加 `theta >= courier_baseline - saving_upper(x)`。该上界必须覆盖完整路径域，尤其 K=2 的连续任务、空返和共享资源；某个受限路径池得到的最大节约值不是这种上界。

### 2.3 对偶退化扰动：有条件的强化，不能偷换锚点

§4，Lemma 4.1、Proposition 4.2，p.31：关闭设施使固定点的若干坐标为零，导致对偶目标中的对应系数为零，出现许多等价对偶解。作者将零坐标改为小正数 epsilon，以偏好较小的对偶系数。其结论明确要求 epsilon 足够小，使扰动问题的最优对偶解仍对原问题最优；用扰动目标还原原锚点值时还需扣掉 epsilon 对应项。

对 MLND 可迁移的是“选点/对偶退化会影响割覆盖面”，不是直接复用 epsilon 数值。我们战略变量有容量等级、设施开闭、分配、航线等联动约束，逐坐标把零改成 epsilon 可能违反它们。合法做法可以在已证明的战略凸包内用内点与当前点凸组合；生成普通割仍须通过原 SOCP 对偶、完整定价及数值证书。仅有全局有效性，不代表该割在原点紧、属于 Pareto 割或满足 Magnanti–Wong 最优面约束。

### 2.4 根节点调度比“总下界数值是否闭合”更具体

§5.1，Algorithm 1，p.32：单树算法在根节点对分数点和整数点均可分离；其他节点只在整数候选处分离。每次按该点是否违反覆盖条件决定加割。§5.2，p.32–33：分数根节点可使用选择性分离阈值，控制割数量和成本；整数可行性条件仍必须满足。

当前代码中，完整连续模型已经求得最优值，只说明取得了一个很强的**标量下界**。将它作为 `F(x)+theta >= LB_root` 一行放入主问题，并不能保证其它战略点上的 `theta` 已接近真实 `Q(x)`。因此不能仅因标量连续界闭合，就认为根节点投影分离已完成。此处可迁移的是继续检查主问题当前点的 `Q(x)-theta`、通过有效割补足投影；不能保证这些割必然提高已知连续模型的最优值。

论文同时比较单树与多树，但当前 K=1 日志中 576 次主问题调用的节点数均为零，因此“保留一棵大树”尚不是现有数据证明的首要收益来源。优先改善候选点上的运营低估更有针对性。

## 3. Bayram et al. (2023)：最大流最小割的准确边界

正式版本地路径：`/Users/songtaowang/Zotero/storage/XGY8CJ5Q/Bayram et al. - 2023 - Hub Network Design Problem with Capacity, Congestion, and Stochastic Demand Considerations.pdf`。核对 §4.1–4.3，pp.1283–1285，PDF 第 8–10 页。

主问题的二元变量为枢纽容量等级 `y_h^ell`，并为每个需求场景保留运营代理 `Omega(s)`；子问题包含该场景的路径和拥堵。它要求需求被路由，容量不足时子问题可能不可行。多个场景代理来自场景独立性，不能直接用来证明当前模型可以任意按 OD 拆成多割。

§4.3.1–4.3.2 给出例子，说明总容量大于总需求也不保证可行。§4.3.3 明确指出，共享容量的多商品流不能直接套用单商品最大流最小割。作者新增按目的地划分的容量预留变量 `z_tilde[h,i]`，把同目的地 OD 聚合成一个超级商品，并要求所有目的地预留容量之和等于已开枢纽容量（式42–43）。

Proposition 1、式44–45 在论文定义的特定图连通性条件下，使每个超级商品的全部最小割条件与相应容量可行性等价；不满足该连通性条件时，这些条件仍可有效但不充分。这里论文的图连通性符号与本项目的 `K=1/2` 连续接单次数无关。

因此对当前模型的结论是：

- 可以研究拓扑和容量导出的必要条件或运营成本下界；不能声称一个单商品最大流模型已经完整刻画共享容量多商品运营。
- 原论文靠新增容量分配变量解决耦合，不是无变量的任意 min-cut；当前“保持精简主问题”的要求下，优先研究将必要关系投影为现有战略变量上的有效割。
- 当前允许骑手兜底，原论文“所有需求必须过枢纽”的可行性割不能照搬；需要成本/节约上界形式的推导。

## 4. 严格 TS 新论文能支持什么

Li et al. (2026) 的出版社摘要确认其是两阶段随机规划，并用受 partial Benders 启发的重构，将部分子问题变量与约束移入主问题，再结合有效不等式。全文未读到，具体保留结构与割式仍待核对。它目前只能作为待查候选，不能作为本轮增加某种变量或场景方法的直接依据。

Silva et al. (2026) 的已读作者版本是 [arXiv:2405.02439v1](https://arxiv.org/html/2405.02439v1)，题名仍带 Single，版本日期 2024-05-03。§5.1 式(7)保留各时段位置变量与按客户的值函数代理；客户子问题独立、为特殊网络流。Algorithm 2 利用结构解析构造最优对偶割，并在整数候选回调加入。作者明确不声称解析过程产生更强的割；它的优势主要是统一结构并降低求解开销。当前 MLND 的共享容量、拥堵和连续任务破坏该独立结构，不能直接套用按客户多割或解析公式。未在这个已读版本看到可直接迁移的 in-out/core-point 稳定化方案；也不将预印本性能倍数当作正式版结果。

## 5. 本轮建议与可验证指标

优先级一：在保持精简主问题的前提下，恢复以点上运营低估为对象的根节点投影分离；选点策略与普通有效割、Pareto 认证分开记录。分数点上的运营解不能成为整数上界。观察根分离次数、实际 `Q(x)-theta`、有效割在整数候选上的违反量、全骑手运营候选比例、每秒获得的有效分离、以及总预算内整数 gap；不能仅以已知连续最优值是否上升判断成败。

优先级二：强化航线/服务可达性对运营节约的上界，而非删除合法的全骑手设计。任何新容量/最大流割先在微型 K=1/2 全战略枚举及完整路径参考上验证，检查所有真实运营解都满足；还要分别测试共享枢纽、缺失航线、连续接单共用资源的反例。

文献支持的是上述机制和审查原则；改善幅度需要当前算例对照实验才能确认。solution-pool 提取时间预留属于当前实现的预算修复，不应包装成来自上述论文的新算法定理。

## 6. 可复核来源

- 本地三份正式/作者 PDF 只读，未改动；Calamita 与 Bayram 的公式、页码核对依据本地全文。
- [Calamita 作者预印本](https://arxiv.org/abs/2401.12625)；正式版以本地 EJOR PDF 为准。
- [Li 出版社页面](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0180)，摘要层证据，2026-10-21 访问。
- [Silva 正式版](https://pubsonline.informs.org/doi/10.1287/trsc.2025.0104)与[作者预印本全文](https://arxiv.org/html/2405.02439v1)，明确版本差异，2026-10-21 访问。
- [Wu 官方复现库](https://github.com/INFORMSJoC/2023.0367)，用于确认期刊与 DOI，不以代码仓库代替未读算法证明。
- [Transportation Science 的 Crossref 元数据接口](https://api.crossref.org/journals/0041-1655/works)，有限年份及关键词候选检索，2026-10-21 访问。
