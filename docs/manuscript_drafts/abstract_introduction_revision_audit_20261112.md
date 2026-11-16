# MLND 摘要与第一章修订记录

日期：2026-11-12。目标文件：/Users/songtaowang/Documents/MLND/Multi-Type Facility Location for Drone-based Pickup and Delivery/main.tex。
范围：通读现有文章，改写摘要和第一章正文；保留原图、标题、关键词、第二章以后正文、既有注释稿及参考文献库。
当前为作者审核稿；文献由代理核读，不声称已经完成人工核验或可直接投稿。

## 修订逻辑

摘要约 200 词，引言约 900 词。
围绕业务决策、服务架构、成本及资源权衡、最接近文献、研究问题、方法和贡献组织。
不再以一般低空行业背景、美国监管例子或企业尚未盈利作为主要论据。
保留经过核查的美团常态化运营事实；没有借用其他论文的数值结论。

参考已读顶刊原文的论证结构：He et al. (2017, M&SOM)、Wang et al. (2022, M&SOM)、Boutilier and Chan (2022, M&SOM)、Bayram et al. (2023, Transportation Science)、Lejeune and Ma (2025, Operations Research)、Wang et al. (2026, Management Science)。
最接近的 Chen/Liu 工作按核实到的预印本版本处理，没有称为顶刊正式论文。
新引言所用文献条目均已存在于原文献库，不新增重复条目。
Chen 的引言引用统一使用文献综述采用的 chenCoordinatedDroneCourier2025。

## 全文核对发现

以下行号为本轮修订前快照中的位置。

| 问题 | 依据 | 本次处理 |
|---|---|---|
| 摘要称先分配再重构需求 | 摘要77行 vs 919、1054行 | 改为网络优化之前重构原始 OD 需求 |
| 摘要声称指数分布骑手时间 | 摘要79行 vs 484–485、767行 | 删除无正文依据的分布限定 |
| 摘要将均匀续单提前量写成剩余飞行时间 | 摘要79行 vs 1100–1108行 | 从摘要移除此层技术细节 |
| 新模型已有性能、规模、案例结论的暗示 | 引言179–181行 vs 1629–1631行 | 删除，摘要/引言不引用旧换电模型结果 |
| 新旧模型的实验未统一 | 1629–1631、1798–1802行 | 留待现模型实验章更新后补入数值发现 |
| 旧附录证明与当前续单重构不一致 | 2084–2203行 | 记录，未擅自重写正文模型或证明 |
| 首创性容易扩大 | Chen/Liu/Bayram 原文 | 明确继承一般分解方法，贡献落在服务架构、资源核算及结构割 |
| 精确分离的范围 | 当前 Appendix C.1 | 限定为构造的 EPI 割族，并写明逐基地排序 |
| 状态下界的成本口径 | 当前 Appendix C.2 | 明确为认证总成本下界 |

仍待论文后续工作：更新第六章当前模型结果、填写数据与算例构造、处理旧附录及其 14 个已有失效引用键。
本次未跑实验，也未把外部保存的旧结果挪入摘要。

## 证据链

- intro_literature_closest_20261112.md：四篇最接近文献，含具体页节与版本。
- intro_literature_structure_20261112.md：四篇顶刊结构参照，含原始全文定位。
- Wang et al. (2026)：本地原文 /Users/songtaowang/Zotero/storage/IWB56XX8/Wang - Does the Locker Alliance Network Improve Last Mile Delivery Efficiency.pdf，pp.4124–4127；出版商 DOI 10.1287/mnsc.2024.04567。
- Dong et al. (2026)：出版商索引公开的引言与摘要，DOI 10.1016/j.trb.2026.103449；未声称读到全文 PDF。
- 美团：https://www.meituan.com/news/NN260522185005269，正文发布日期为 2026-05-21。
- intro_abstract_sources_20261112.json 与 intro_abstract_claims_20261112.csv：证据标识、逐句哈希及核查状态。

## 整合与检查结果

已直接写入原 main.tex，摘要和第一章修订均为蓝色，每句独占源码行。
桌面编辑器 compile_latex_document 返回 success。
新增引用键均可解析，没有新增失效交叉引用；全文原有 14 个缺失引用键保持不变。
原图、第二章以后内容、历史注释稿和两个现有 BibTeX 文件保持原样。
本次只确认本地稿件及编译，未验证 Overleaf 云端同步。
修订后源文件 SHA-256：b6454c0ae4082dc52adc09afc4d90119e1a26cb6a798d8a89575f1a5904fa631。
修订前备份：/private/tmp/mlnd-intro-revision-20261112-ta7zxwn6/main.before.tex。
