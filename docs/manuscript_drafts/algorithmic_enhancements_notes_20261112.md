# Algorithmic Enhancements 英文草稿：结构与证据

状态：供作者审核的独立草稿，2026-11-12。本次未改动用户附件、正式 manuscript 或 Better BibTeX 管理的文献库，也未运行数值实验。当前打开的英文文件已补齐独立预览所需结构，接入已有附录及四篇文献，并通过桌面编辑器编译；这不表示正式 manuscript 已整合编译。

## 文件与放置

- `algorithmic_enhancements_en_20261112.tex`：当前编辑器中的完整可编译文档；`BEGIN/END MANUSCRIPT SUBSECTION` 标记之间可替换现有 Algorithmic Enhancements 小节。保留 Cut Strengthening / Warm Starts 层级，两类割采用文献已有名称 Extended polymatroid inequalities 和 Disjunctive cuts；每个英文句子独占源文件一行。
- `algorithmic_enhancements_appendix_en_20261112.tex`：放在已有 `\appendix` 后，包含队列成本下界推导、次模性、EPI 有效性与排序分离证明、任务状态割证明及简短方法补充。主文引用的所有新增附录标签均已提供。
- `algorithmic_enhancements_20261112.bib`：四篇引文的补充条目。键名是本草稿键名，整合时通过既有文献管理流程导入或映射，避免直接覆盖 MLND.bib。

正文保留五个展示公式：任务节约上界、队列集合函数、其与原运营成本的联系、EPI、任务状态析取割。
前两者定义系数与结构，第三个连接原问题和被凸化的下界，后两个是加入主问题的约束。
状态下界的精确 inf 定义移至附录，正文用明确的“其余设计自由、总成本的认证下界”说明。
正文不展开消去可用率的优化、边际递减和分部求和证明、完整域认证与列生成修正、ALNS 操作流程。
Warm Starts 为两句，只列列池复用、incumbent-based MIP starts 和明确限定 L2/K2 的 ALNS + MISOCP repair。

## 主张与依据

以下为工具辅助核对记录；不声称已经完成作者的人工作者审核或投稿批准。

| ID | 主张 | 依据与定位 |
|---|---|---|
| C1 | 当前逐任务流量、机队约束和非负成本给出队列成本下界 | 用户 TeX 的 SP、idle、fleet 约束；`docs/queue_submodularity_propositions_zh_20261105.md` 命题1；本稿附录重新按任意可行运营解、最小化rho、再取inf的顺序推导 |
| C2 | 凹函数复合非负模函数产生次模结构 | `docs/epi_literature_20261112.md`；Yu–Küçükyavuz2023 作者稿§2.1–2.2；本稿附录给出模型专用证明 |
| C3 | EPI有效且降序排列精确分离固定割族 | Atamtürk–Narayanan2022 作者稿§2、Remark2和式(6)，https://arxiv.org/pdf/1912.13238 ；本稿附录给出自包含证明 |
| C4 | 状态总成本下界构造析取割 | `docs/design_face_literature_20261112.md`；Chen–Luedtke2024 作者稿§2.2–2.3和式(15)，https://optimization-online.org/wp-content/uploads/2021/11/8687-1.pdf ；Kronqvist–Misener2021 出版正文§3、式(13)、Theorem1，https://doi.org/10.1007/s11081-020-09551-6 |
| C5 | warm starts与ALNS的实现范围 | `cpp/src/benders.cpp` 的 pool_p/pool_e、master add_start；`cpp/src/linear_model.cpp` 的 addMIPStart；`docs/current_baseline.md` 的 L2/K2 范围与 `docs/task_assignment_migration_20261105.md` 的独立任务语义 |

参考应用论文的行文顺序是“模型中的特定结构—必要定义—有效不等式—分离/嵌入方式”，对应 Hu等2021 的§3.5，定位与访问边界见 `docs/cut_application_examples_20261112.md`。
正文仅引用直接理论来源；不将 Hu 的应用函数、Chen 的计算结果或通用割构造的原创性归于本模型。

## 已核查的范围

- 队列集合下界明确只先对整数设计建立；线性 EPI 随后用于主问题分数点分离。
- 单订单与双订单骑手成本分别定义，同 OD 双订单计两次；零率/无路径系数为零。
- 不把完整 Q(d) 称为次模，也不声称任意有效割中的全局最强性。
- 设计面针对含固定费用的总成本，其余设计变量自由，不同任务的总成本界不相加。
- 完整定价认证与局部修复的上下界角色在附录区分；没有声称稳定计算提速。
- 已检查新增标签、交叉引用、引文键和 LaTeX 环境配对；完整独立文档已通过桌面编辑器编译。

本稿只展开正文选定的两类不等式；资源价格扩展及双任务联合构造继续保留在既有理论与实现文档中，未把它们误写成已经由本附录完整覆盖。
