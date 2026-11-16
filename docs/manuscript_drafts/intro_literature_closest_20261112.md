# MLND 摘要与引言：四篇最接近文献的证据笔记

核对日期：2026-11-12。范围：书目信息、研究定位、摘要与引言组织；不修改论文、BibTeX 或实验。检索仅使用公开题名，未向外部服务发送未公开稿件。

已读取四篇正文，而非仅摘要：Chen 的 arXiv 全文；Liu 的 arXiv HTML 与本地 PDF；Wang、Bayram 的本地期刊全文，且分别用出版社和机构仓储核对出版信息。下文“我们的差异”“写法建议”是对现有 MLND 模型的比较判断，不是被引文献替本研究证明的结论。

## 1. Chen：最接近的系统与队列建模文献

Shuiwang Chen, Kai Wang, Lingxiao Wu, Wei Qi (2025). *On Coordinated Drone-Courier Logistics for Intra-city Express Services*. arXiv:2501.05200v1；DOI 10.48550/arXiv.2501.05200。本次核实版本为预印本，不能称为已发表 M&SOM 论文。建议使用现有键 `chenCoordinatedDroneCourier2025`；`chenCoordinated2025` 是同一文献的另一个已有键，不应当成两项研究。[版本与摘要](https://arxiv.org/abs/2501.05200)，[作者全文](https://arxiv.org/pdf/2501.05200)。

- **已读定位：**摘要 p.1；§1.1 pp.2–3；§1.2 pp.3–4；§2.3 pp.6–7；§3.1 pp.7–9，页码均为预印本自带页码。
- **对象与方法：**快递员承担首末端，无人机在 vertiports 间运输；设施规划与闭合队列、机队、调拨和包裹集中运输耦合。模型为非凸 MINLP，以保守/松弛模型间的自适应分段线性逼近收敛上下界。§3.1 明确同一 vertiport 对之间按 pooling size Q 集货出发，不能说其不考虑订单合并。[全文 §1.2、§3.1](https://arxiv.org/pdf/2501.05200)。
- **组织方式：**摘要按问题障碍→协作结构→模型及算法→案例发现展开；引言先解释末端交付障碍及系统结构，再提出三个实践问题，最后按模型、算法、实证逐项贡献；详细文献综述另设 §2。这比开头立即列全部约束更适合作为我们的骨架。[全文 pp.1–4](https://arxiv.org/pdf/2501.05200)。
- **可辩护差异：**我们的基地承担机队支持、储物柜承担交接，两者可分布在不同地点；完整活动还包括出库、连续两单服务和返航。应比较这种功能分离及资源核算，不能把“无人机—快递员协作”或“队列联系战略与运营”单独称为新贡献。其同航线多包裹 pooling 与我们的连续执行两单不是同一机制。

## 2. Liu：最接近的餐饮配送与异质设施文献

Yang Liu, Yitong Shang, Sen Li (2025). *Joint Infrastructure Planning and Order Assignment for On-Demand Food-Delivery Services with Coordinated Drones and Human Couriers*. arXiv:2501.14325v1。现有键 `liuJointInfrastructure2025`；本次未核验到正式期刊版本。[版本与摘要](https://arxiv.org/abs/2501.14325)，[作者 HTML 全文](https://arxiv.org/html/2501.14325v1)。

- **已读定位：**摘要 p.1；§1 pp.2–4；§2 pp.4–6。HTML 的 §1 含相同贡献段。
- **对象与方法：**同时选址 launchpads、kiosks，并分配地面/空中配送；地面订单 bundling、骑手匹配均衡、起降点双端队列进入 MINLP，神经网络近似相关非线性映射后嵌入 MILP。§1 明确空中 launchpad→kiosk 段不 bundling，末端仍由快递员完成。[全文 §1](https://arxiv.org/html/2501.14325v1#S1)。
- **组织方式：**摘要先明确业务流程与决策，再交代方法和案例中的时间权衡；引言由需求背景→地面运营困难→无人机在密集城区的交付限制→协作业务流程→模型、算法、发现三项贡献。其背景较长，MLND 可以借鉴“障碍如何导致系统设计”的连接，不必照搬篇幅。[全文摘要、§1](https://arxiv.org/html/2501.14325v1)。
- **可辩护差异：**“两类设施”“基础设施与订单分配联合决策”已有直接先例。我们应强调同一储物柜兼具取件/交付用途、基地与交接设施分离，以及自提/送货选择、连续任务与舱位占用在同一规划模型中的联系。不能说现有协作模型只有单一设施类型。

## 3. Wang：已发表顶刊的摘要与引言组织参照

Kai Wang, Alexandre Jacquillat, Vikrant Vaze (2022). *Vertiport Planning for Urban Aerial Mobility: An Adaptive Discretization Approach*. **Manufacturing & Service Operations Management 24(6):3215–3235**；DOI 10.1287/msom.2022.1148。现有键 `wangVertiportPlanning2022M&SOM`。[出版社](https://pubsonline.informs.org/doi/10.1287/msom.2022.1148)，[作者出版目录](https://mitmgmtfaculty.mit.edu/ajacquillat/research-publications/)。

- **全文核对位置：**期刊 pp.3215–3217 的摘要、§1 和 §2；本地文件为 `/Users/songtaowang/Zotero/storage/BQRWYSC8/Wang et al. - 2022 - Vertiport Planning for Urban Aerial Mobility An Adaptive Discretization Approach.pdf`。
- **对象与方法：**面向载人 UAM，设施位置/容量、车辆运营和内生乘客需求共同决定盈利；队列与设施部分可作 MISOCP，非凸需求关系用自适应离散化处理。并非餐饮交接系统。[出版社摘要](https://pubsonline.informs.org/doi/10.1287/msom.2022.1148)；期刊 §1–2。
- **组织方式：**摘要按问题→数学困难→方法→计算结果→管理含义展开。引言 p.3216 先明确“规划—运营”与“运营—需求”两类相互依赖，再列三个实践问题；随后按模型、算法、计算、管理发现四项贡献展开，并把结果回扣到前述问题。正文 p.3217 主动说明继承 He 等哪些思想，再说明哪些结构不同。
- **可辩护差异：**我们是给定规划需求及其固定任务重构下的配送成本最小化；不能借用其利润、内生市场渗透或全球非凸优化结论。可借鉴其“先讲相互依赖为何造成决策困难，再讲模型如何回应”的论证顺序。

## 4. Bayram：最接近的分解算法先例与紧凑写法

Vedat Bayram, Barış Yıldız, M. Saleh Farham (2023). *Hub Network Design Problem with Capacity, Congestion, and Stochastic Demand Considerations*. **Transportation Science 57(5):1276–1295**；DOI 10.1287/trsc.2022.0112。现有键 `bayramHubNetwork2023TransportationScience`。[机构出版记录](https://kar.kent.ac.uk/101333/)，[作者接受稿](https://kar.kent.ac.uk/101333/1/TS-2022-0112_Main.pdf)。

- **全文核对位置：**期刊摘要及 §1 pp.1276–1277，§2 p.1278；接受稿对应 pp.1–4（仓储 PDF 另含一页封面）。本地期刊版在 `Zotero/storage/XGY8CJ5Q/`。
- **对象与方法：**随机需求下同时选择枢纽、容量及事后路径；以拥堵成本连接规划与运营。明确已有路径型 MISOCP、SOCP 对偶 Benders 分解、列生成和主问题强化。[作者稿 §1 pp.3–4](https://kar.kent.ac.uk/101333/1/TS-2022-0112_Main.pdf)。
- **组织方式：**非结构式摘要紧接问题扩展、决策、数学模型、求解方法及发现。引言用“流量集中带来收益，也造成拥堵”的具体矛盾，推出容量与路径必须联合决策；再解释传统两枢纽路径为何不足，才引出路径模型和分解算法，最后提出实践问题。[作者稿 §1](https://kar.kent.ac.uk/101333/1/TS-2022-0112_Main.pdf)。
- **可辩护差异：**我们的难点来自有限无人机循环、完整单/双订单活动、交接选择和储物柜占用。Benders+CG 的组合已有直接先例；贡献应落在本模型的路径与资源结构、有效下界构造及相应求解实现，不能仅靠组合名称主张算法新颖性。

## 5. 对重写的建议（编辑判断）

1. **摘要：**用具体规划决策开篇；随后以一两句交代设施功能分离为何同时影响地面交接、无人机循环和容量；再写统一路径型 MISOCP、Benders/列生成及强化；末尾只能给当前模型已验证的计算发现。尚未完成的数值研究不应写成确定结论。
2. **引言：**业务障碍→拟研究的设施与服务结构→一个明确的联合决策问题→与 Chen/Liu 的精准比较→模型和方法贡献。Wang/Bayram 用于支持“为什么必须把规划与运营相连”及方法定位；无需在引言提前重述整个文献综述。
3. **差异不写成特征清单。** 将“基地/交接分离”连接到额外空载移动和任务路径，将“交付选择”连接到舱位占用和地面成本，再说明为何独立选址或仅比较空中飞行时间不足。以上是本模型的解释，若进一步声称某结构提高效率或改变最优布局，仍需本研究结果支持。
4. **保留边界：**不宣称第一项协作配送研究、第一项异质设施联合规划或第一项 Benders+CG；不把固定任务重构说成在线匹配最优策略；不把其他论文的集中布局、盈利或时间节省结果写成本研究发现。
