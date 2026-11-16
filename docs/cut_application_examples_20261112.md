# 割平面应用文献与写作范例核验

核验日期：2026-11-12。范围为 2020—2026 年的选址与网络设计应用；只记录本轮实际读到的原始来源。E1 指凹函数复合非负模函数所得次模上图的扩展多面体不等式（EPI）；E2 指互斥离散状态下的成本下界聚合。这里区分“同一数学割族”与“写作或结构类比”。

## 1. 最直接的已核验应用范例：Hu et al. (2021)

**完整书目信息。** Qing-Mi Hu, Shaolong Hu, Jian Wang, Xiaoping Li. *Stochastic single allocation hub location problems with balanced utilization of hub capacities*. Transportation Research Part B: Methodological, 153, 204–227. [DOI](https://doi.org/10.1016/j.trb.2021.09.009)。

**原始来源与定位。** [作者机构提供的期刊 PDF](https://faculty.swjtu.edu.cn/system/resource/tsites/download.jsp?ap=lwyfj&fid=EA52E74830194C54B43CC9A9F1B40214&fl=Bvma2uueqmim&uid=12015)：§3.5 “Valid inequalities”，期刊页 **214**；该页包含式 **(65)**、Definitions **1–2**、Theorem **3**及其证明。直接下载目前返回反盗链提示，但机构 PDF 的搜索索引完整呈现了这一个正文页；本条定位与公式均据该页核验，不是据摘要推断。

**应用机制。** 随机单分配枢纽选址模型包含

\[
V'_k\geq\sqrt{\sum_i(\sigma_i^*)^2x_{i,k}^2} \tag{65}
\]

利用二元性，作者得到集合函数 \(g(A)=\sqrt{\sum_{i\in A}(\sigma_i^*)^2}\)。Theorem 3 用某个排列的前缀集合构造边际增量 \(\pi\)，证明 \(\sum_i\pi_i x_{i,k}\leq V'_k\) 有效：\(\pi\) 是相应扩展多面体的极点，故对任意整数支撑集有 \(\pi(A)\leq g(A)\leq V'_k\)。该页随后指出可用 Edmonds 贪心算法寻找这些割，并将加入割的两个重构模型分别标记为加 CUTS 的版本。

**对应关系与写法。** 这是 **E1 的直接应用类比**：同为二元变量上“凹函数复合非负线性和”的上图强化。它不涉及我们的具体排队函数，也不证明 E2。值得参考的段落顺序是：指出原模型中的具体非线性约束 → 识别次模结构 → 给出有效性命题与简短证明 → 交代分离与算法嵌入。无需先展开泛化的加速策略综述。本轮不摘录其数值提升幅度。

## 2. 较新的直接应用证据：Malik and Vidyarthi (2025)

Aditya Malik, Navneet Vidyarthi. *Fulfillment Center Location and Network Design in Dual-Channel Retailing*. Networks and Spatial Economics, **25**, 331–365；在线发表日期为 **2024-08-28**，卷期年份为 **2025**。[出版社原始页面](https://link.springer.com/article/10.1007/s11067-024-09647-x)，[DOI](https://doi.org/10.1007/s11067-024-09647-x)。

出版社摘要明确说明：模型联合决定履约中心选址、线上履约门店、门店分配与安全库存；将 MINLP 重构为 MISOCP，并在精确分支割算法中分离 EPI。因而可用于支持“近期选址与网络设计研究仍采用 EPI 强化”的陈述。

**证据边界：本轮只核验出版社摘要及书目信息，未获得期刊正文。** 不提供猜测的节号、公式号、定理号或计算改进幅度；也不能由摘要判断其 EPI 的全部具体系数构造，更不能当作 E2 的依据。适合作为补充应用引用，不宜充当主要证明或章节写法模板。

## 3. 有完整全文的近期补充：Qian et al. (2026，预印本)

Weikang Qian, Keyan Li, Wei-Kun Chen, Yu-Hong Dai. *Polyhedral results for two classes of submodular sets with GUB constraints*. arXiv:2601.18360v1，2026-01-26。[作者预印本全文](https://arxiv.org/pdf/2601.18360v1)，[HTML 全文](https://arxiv.org/html/2601.18360v1)。本轮未核验期刊录用状态。

精确定位：§1.1、PDF 页 **2–3**、式 **(3)–(4)** 将带设施类型互斥约束的最小概率覆盖选址问题写成凹函数上图；§3 构造提升 EPI；§5、页 **18–22** 处理凸包与精确分离，Corollary **5.5** 给出 \(O(n\log n)\) 分离；§7.1、页 **25–26** 及 Table **2** 比较该选址问题中的 EPI 与提升 EPI。

**关系。** 这是 E1 加入 GUB 互斥结构后的直接理论与应用参考。每个设施位置最多选一个类型，与任务最多选择一个出发基地有结构相似性。但其提升 EPI 依赖特定凹复合函数，**不是 E2 中任意状态成本下界表的聚合**；不能把互斥结构相似性写成两种割等价。分离复杂度也只对文中定义的割族成立。

## 本章可采用的引用安排

正文构造 E1 时，先给出本研究自己的排队函数、定义域、有效性及排序分离证明，再引 Hu et al. (2021) 作为应用先例；Malik and Vidyarthi (2025) 可并列作为较新的网络设计应用。Qian et al. (2026) 更适合讨论互斥结构下进一步强化的相关研究，且应明确预印本身份。

本轮未找到并核验一个与 E2 完全一致的独立选址应用范例。不要用竞争选址中最大化次模收益的上界割来替代 E1 的上图割，也不要将 GUB 提升 EPI 当作 E2 的证明。E2 可另用已核验的稀疏割／状态分解理论来源说明，再直接给出本模型的构造。

写作时将“有效”“在该割族中分离最强”和“实测改善运行时间或全局下界”分开陈述：前两项需各自的数学论证，最后一项只由本研究的受控计算结果支持。
