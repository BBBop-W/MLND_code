# L2 上界元启发式完整方案：资源块多起点 ALNS 与原模型修复

日期：2026-10-30。状态：**原始设计记录，保留设计时的接口与阶段表述**。策略主流程现已实现，功能验证通过，生产性能尚待对比；具体实现、测试与边界见 [实现记录](l2_primal_alns_implementation_20261030.md)。适用范围为 L2、K2、固定预处理任务率、origin assignment、aggregate flow、无 first-leg consistency。H40 暂不纳入个人电脑测试。

建议采用“**资源块多起点构造 → 自适应破坏修复 → 少量精英精修**”。搜索对象是共同承担固定费用、容量和返回平衡的一组任务与资产；连续流量、共同可用率和队列由原模型的 SOCP/MISOCP 求解器解码。算法输出随时可用的可行上界，认证下界仍由 Benders 主流程负责。

**1. 本方案要解决的具体问题**

同一 R03、600 秒测试中，旧算法检查点已独立核验的最好 UB 为 1327.059162，新下界版为 1324.604929，CPLEX 为 1316.533353。旧最终文件的 1386.160631 是最佳解保存缺陷，不能拿它夸大上界算法空间。新版定价和加列约占其已归集子问题时间的 88.8%，因此每个邻居都完整 CG 不可行。数据与验证见 [运行后分析](/Users/songtaowang/Documents/MLND_code/outputs/lb_cert_upgrade_20261030/postrun_analysis/ANALYSIS.md) 和 [检查点验证](/Users/songtaowang/Documents/MLND_code/outputs/lb_cert_upgrade_20261030/postrun_analysis/baseline_checkpoint/native_validation.json)。

已有 `solve_primal_seed_portfolio` 的 H20 前置调用没有实际 repair：逐 head 正收益入口与最多 6 个 locker 的候选限制，使候选域可能为空。已有好解的部分 head 支持使用 9–11 个 locker；这不证明 6 个 locker 必定不可行，却证明目前构造规则无法表达这些已知好解。此前诊断见 [上界诊断](/Users/songtaowang/Documents/MLND_code/docs/l2_ub_postmortem_20261030.md)。

本方案的三个直接目标是：从 all-courier 独立启动；允许联合跨越固定费用门槛；用大量便宜且可行的评价替代少量昂贵的全域精确评价。是否能追平或超过 1316.533353 必须由后续同预算实测回答，该值也不是已知最优。

**2. 文献依据与本项目设计的边界**

下表来源于本次在线复核的一手论文/作者稿。文献支持方法构件，不证明迁移后的 MLND 收益。下文参数、资源闭包、预算及具体算子均为针对本模型的设计选择。

| 一手来源与本次读取范围 | 可采用的机制 | 不移植的结论 |
|---|---|---|
| Ropke–Pisinger（2006）；[DTU 作者预印本](https://backend.orbit.dtu.dk/ws/portalfiles/portal/3154899/An%2Badaptive%2Blarge%2Bneighborhood%2Bsearch%2Bheuristic%2Bfor%2Bthe%2Bpickup%2Band%2Bdelivery%2Bproblem%2Bwith%2Btime%2Bwindows_TechRep_ropke_pisinger.pdf)，重点 §3.3–3.5 | 多种破坏/修复竞争、历史收益更新权重、可接受较差可行解以扩大搜索 | 原取送路径的删点插点和时间窗编码不等于本模型的任务率/资源决策 |
| Hendel（2018），[作者全文](https://optimization-online.org/wp-content/uploads/2018/12/6992.pdf)，重点 §2–3 | 有预算的邻域 MIP，动态控制规模，按收益与代价选择邻域 | SCIP MILP 实现及性能不能直接视为当前 CPLEX MISOCP 的性能 |
| Fischetti–Lodi（2003），[Local Branching 原出版页](https://link.springer.com/article/10.1007/s10107-003-0395-5)，出版社摘要 | 外层控制邻域，通用 MIP 求解器探索局部设计子域 | 本方案仅取上界搜索部分，不继承其完整精确框架的最优性保证 |
| Danna–Rothberg–Le Pape（2005），[RINS 原出版页](https://link.springer.com/article/10.1007/s10107-004-0518-7)，摘要/元数据 | incumbent 与松弛点的差异可指导释放变量 | 当前 master 的松弛点只作 RINS-inspired 指导，不能冒充原全模型 LP 点或永久固定全局变量 |
| Hewitt–Nemhauser–Savelsbergh（2013），[网络设计作者稿](https://optimization-online.org/wp-content/uploads/2012/02/3360.pdf)，§1–4；[正式 DOI](https://doi.org/10.1287/ijoc.1120.0503) | 用网络结构和列信息构造较小整数限制问题，并围绕好解搜索 | 其完整 branch-and-price 证明不适用于我们的受限路径 repair |

ALNS 的自适应权重和模拟退火接受机制有上述来源；本文使用较少资源块算子、按实际秒数计价、按完整 head 状态而非所有航线位计距离，是 MLND 的具体设计。没有复制文献参数或声称其是本实例的最优参数。

**3. 状态表示、目标函数与必须保留的模型语义**

一个搜索状态包括战略设计与一份已验证运营解：

- head 状态 `a[r] ∈ {0,1,2}`，分别为未分配、基地 1、基地 2；同一个 head 的全部正率 single/pair 分支共同使用该出发基地。
- 基地开启 `y[l]`；locker 状态 `k[h] ∈ {0,…,n_levels}`（0 为关闭）；航线开启 `z[u,v]`。航线由候选路径资产并集产生，交给 repair 选择，不逐位盲目随机翻转。
- 连续运营解包含共同 `rho[l]`、service/empty flow、head、courier、idle、fleet 等原变量。它们不进入随机编码，始终由原模型求解并验证。
- 物理支持保存精确路径 ID/签名、严格正流路径和实例/任务率/目录身份。成本使用完整原目标，不用代理分数或简化运营成本更新 UB。

令正率分支集合为 `B_r = {single: ν_r^S>0} ∪ {s: ν_rs^P>0}`。在整数分配下，原乘积约束给出 `v_rl = α_rl ρ_l`，每个分支满足 `Σ_{p∈P_rlb} f_p = ν_rb v_rl`。因此若 `α_rl=1` 而一个正率分支缺列，就会强制 `ρ_l=0`，影响同基地全部 head。完整分支覆盖是产生正服务候选的必要条件；覆盖本身不保证队列、容量和返回平衡可行。[现有建模接口](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp)

pair 的第二个 OD 不是另一个需要同步改写分配的 head；不改变固定任务率，不拆分共同 rho，也不删除低率分支。所有原需求守恒、正率分支、locker 容量、设施链接、机队、锥约束及基地流量平衡都保留在 repair 中。未分配/未被无人机覆盖的需求继续走原模型 courier 通道，不增加新罚函数或新服务语义。

**4. 支持库与资源闭包：一次准备，多次复用**

建立按 `(head, departure_base, branch)` 索引的物理路径库。优先复用已有目录、支持图、当前 CG 列池和已验证解的正流路径；不在每次 ALNS 调用中重新扫描全部约 139 万条列。

每个正率分支初始保留 4 条具有不同资产/返回模式的候选：低运营成本、低新增固定费用、低拥挤资源用量、不同返回基地。可在不重复的情况下扩到 8 条；这些是短名单策略，不是全域支配证明。支持库构造中止时标记 `partial`，缺少见证只说明本候选库不足，不宣布原模型不支持该 head。价格/资源评分变化时只重排已存短名单，必要时有预算地补充候选。

每个邻域的列域必须包含：所有保留设计所需的 incumbent 严格正流路径、释放 head 每个可用出发基地的完整正率分支候选，以及可用的平衡/空返列。L2 当前 direct 空返目录很小，可完整包含；不能要求每条跨基地 service 路径必须逐条搭配一条空返，因为不同 service 流本身也可能互相平衡。是否可行由整体 balance 约束决定。

关闭/替换资源时，释放集合按以下闭包构造：

1. 选中一组 head 或一个待关闭资产。
2. 纳入这些 head 的旧正流路径资产和新候选路径资产，释放相应 locker 的**全部容量档位**以及相应航线。
3. 若强制关闭资产，将当前使用该资产的所有 head 加入释放集；关闭基地则还包括所有分配到该基地、或使用其返回/空返设施的受影响 head。
4. 对新增 head 补齐正率分支，再重新扩展旧/新资产支持，直到稳定；若规模过大，换较小的原始破坏块或放弃该次邻域，而非截断必需支持。
5. 未释放 head 的战略分配保持当前值，但其运营流、共同 rho 和机队仍可重新优化。

locker 数量不设置固定 6 个的硬限制；按 repair 变量数/非零元/实际建模时间控制规模。必须保留的正流支持优先于可选列数限额，不能为了满足限额丢弃热启动可行性。

**5. 多起点构造：必须在 served=0 时可执行**

起点从 all-courier 和合法外部 incumbent 开始，先原子保存可行兜底。为 L2 的三个非空基地组合 `{1}`、`{2}`、`{1,2}` 各准备一个起点方向。每个方向改变资源簇和 head 组合，不只是改变基地过滤器：

- 共享固定费用方向：以一个 locker/航线簇为核心，选择候选路径资产重叠较高的 head。
- 高节省方向：按完整分支的骑手成本节省潜力排序，在前 20% 的候选中随机选择，再补齐共享资源。
- 双基地平衡方向：偏好具有互补 service 返回方向和低额外空返成本的 head 组合。

用宽度 8 的小 beam，先成组扩展 2、4、8 个 head，比较**组级**代理收益；允许单个 head 或中间步骤代理收益为负。候选拥有完整分支见证后，才交给原 MISOCP 同时决定 head、资源和运营；不把代理正收益当可行改善证明。对共享固定费的联合激活不加“每个 head 必须单独盈利”的条件。

首轮总额度 30 秒：支持库/资源图最多 8 秒，至多 3 个非重复候选各最多 6 秒，剩余时间用于原验证及保存。若既有支持库已准备好，节省的时间可用于 repair。未产生非空候选时必须记录 `missing_branch`、`support_partial`、`closure_too_large`、`selection_budget` 等不同原因；这不算一次已执行的 MIP repair。不得在连续根或完整 CG 结束后才给初解构造第一次机会。

**6. 六类资源块邻域**

| 算子 | 释放/强制改变的主要对象 | 模型目的与触发条件 |
|---|---|---|
| 新簇激活 | 未分配或低服务的一组 head，加一组共享 locker/航线；可强制至少一个 head 改变三态 | all-courier 可触发；跨越单个任务不赚钱的固定费门槛 |
| 共享资产重排 | 随机选一个 head，再按共同 locker/航线相关性选邻近 head；释放旧/新资源闭包 | 常规改进与资产共享；按真实使用/候选重叠而非仅地理距离分组 |
| 关闭/交换资源 | 一个低利用率高固定费 locker 或航线，及全部当前依赖它的 head；允许新替代资产 | 强制关闭目标，repair 决定重分配、替代容量或退出无人机服务 |
| 拥堵缓解 | 高占用 locker 的依赖 head、容量档位、替代 locker 和路径 | 不单独增加容量；让容量、分配、共同 rho 与流量联合改变 |
| 基地迁移/平衡 | 一组 head 的完整三态、两个基地及返回/空返资源 | 改变共同 rho 与跨基地循环；单次迁移组须保留所有正率分支 |
| 精英差异重组 | 两份已验证精英解的分歧 head/容量/资产和正流路径并集 | 至少有两份不同解才可用；无 LP 点时也能运行 |

有可信 master 指导点时，可用其与 current 的分歧选择一部分释放组，标记 `master_guided`。这是启发式排名，不作全局固定。一般以 4 个 head 起步，连续 3 次无改善后扩大到 8、再到 16，出现改善后回到 4；强制资产闭包可能超过该数量，须按实际模型额度决定是否执行。

**7. 两种 repair 及三个评价层次**

一次 repair 在原模型上固定未释放战略变量，只包含选定物理列，并保留全部连续变量和原约束。它是原模型的限制问题；其中任意通过原验证的解都是合法 UB，局部 best bound 不能更新全局 LB。

两种 repair 模式必须明确区分：

- **强化 repair**：current 仍属于邻域，保留完整可行热启动，按原成本优化，争取改进。
- **探索 repair**：对选定主 head/容量状态增加至少一次组级改变，或强制一个资产关闭/迁移，以排除当前状态；允许得到稍差的可行邻居。全局 best 单独保留，不能把它作为一个不满足探索限制的“邻居结果”。不加 `objective ≤ best` 的硬截断，否则后面的劣解接受机制根本无法工作。

组距离对一个 head 的 0/1/2 三态只计一次改变；容量关闭/任一档位也只计一次。若当前 head 未分配，改变指标是 `α_r1+α_r2`；若分配到 l，则为 `1−α_rl`。可用这些线性表达式实施“至少一个主组改变”；不按大量航线位计算搜索半径。探索 repair 的原成本上限初始设为 `best × 1.03`，只限制这次探索，不产生全局约束。

| 层次 | 工作与初始预算 | 可作出的判断 |
|---|---|---|
| A：廉价筛选 | 缓存/覆盖/身份/资源闭包检查、启发式评分；每次目标 ≤0.2 秒，计入真实总预算 | 缺物理见证可放弃该候选；代理负分只影响排序；不能认定原模型不可行或无改进 |
| B：原模型局部修复 | 通常 3–6 秒，困难但有希望时最多 10 秒；构造/求解/验证都包含在额度内 | 已验证整数可行方案即可提交 UB；无需该邻域最优或完整定价结束 |
| C：精英精修 | 只给新 best 或有明确多样性的精英；最多两次、各 15 秒，复用现有列 | 可扩大路径域或调用固定设计 CG；只有满足原认证规则的 cut 才交给 Benders |

候选列初始软额度 30,000，单次新增可选列软额度 5,000；保留正流支持和分支覆盖优先。连续三次 repair 中建模时间超过单次额度的 30%，先缩小原始释放块/可选路径数，不删必需列；连续两次在不到一半额度内闭合且无改善，扩大组规模；遇到预算耗尽记为未知，不能记作邻域无改进的数学证明。

固定全部战略状态时可用 SOCP，其余用 MISOCP。solver 内部 gap 初始 2% 只是局部停止条件，不是接受可行 UB 的必要条件，也不是全局 gap 证书。任何通过验证的更好解应在 solver 返回/安全回调时尽早提交；不要为取更多 pool 解或做 cleanup 把它留在局部变量中。

**8. 搜索状态、接受、权重和精英池**

始终区分 `best`（全局最好、单调不增）与 `current`（下一次搜索中心，可暂时变差）。所有参与接受的状态都必须通过原模型验证；当前设计的局部限制不可行时，保留 current，不用罚函数把不满足队列/任务率的点混入可行池。

若新解成本 `C' ≤ C_current`，接受为 current；否则以 `exp(−(C'−C_current)/temperature)` 接受。初始温度取 `0.01 C_best / ln(2)`，即初始阶段对高 1% 的可行解约有一半接受概率；按**已使用的上界模块秒数**做几何降温，预算耗尽时降到初温的 1%。这些是本文初始参数，不是引用论文的原参数。连续 8 次有效 repair 无 best 改善时，从多样性最大的合格精英重新开始，并扩大一次资源块；不清空全局 best。最多允许一次升温至初温的 25%，且仍受探索成本上限约束。

前一轮让每个可用算子至少尝试一次；之后采用带 20% 均匀探索的权重轮盘：`P(o)=0.2/m+0.8 w_o/Σw`。一次调用的基础奖励取互斥的 4（新 best）、2（新 current 改进）、1（接受的新颖可行劣解）、0（其他）。令完整调用耗时为 t 秒，增加至多 1 的实际相对 UB 改善奖励，再除以 `5(1+t/5)`，得到 [0,1] 奖励。被调用算子的权重更新为 `0.8w+0.2reward`，其他权重不变；未验证解、空候选、超时无解只获得 0。失败时间也记入 t，不能通过把昂贵构造时间移出计时而虚增奖励。该时间奖励是针对现有慢评价的设计，并非从文献照搬。

精英池初始容量 8。按 head 三态、容量档位、基地和实际使用航线的规范签名去重；相同设计若得到更低运营成本，则更新原项而不是占据第二个名额。优先保留最好解，再按 head/容量组距离保留不同且成本不超过 best 的 2% 的解；池未满时可放宽到 5%。运营流不能仅以舍入哈希判断可行性，哈希只用于索引，最终比较和验证使用原数据。

缓存分为物理支持缓存、已验证可行解缓存和评价履历三类。键含输入/任务率/模型语义/目录签名及战略设计、局部列集合版本。内部候选尽量保存稀疏路径 ID/流量和不变目录引用，避免每个邻居复制完整 139 万列的 JSON；导出和验证仍统一使用原规范，补零/索引转换不能改变语义，其时间同样计入。一个受限域“最优”或“无解”只能缓存到那个局部域；域扩大后必须失效，不能据此禁止全域搜索。存储 `time_limit`、`construction_incomplete`、`validated_feasible`、`restricted_optimal`、`full_recourse_certified` 等不同状态，不把未知写成失败证明。

**9. 与 600 秒 Benders 总预算的集成**

默认集成版建议上界模块额度为总预算的 30%，即 180 秒：初解 30 秒、常规 ALNS 120 秒、精英精修 30 秒；剩余时间归现有 Benders、支持构造和下界工作。原有 initial-design、受限列 MIP、primal neighborhood 和新增 ALNS/精英精修共用这一个上界额度；初解替换原 portfolio 调度，不再另加原来的 20% 启发式预算。任何未用额度归还主流程；不额外赠送初始化时间，不同时启动第二个求解线程。全局停止仍以整个求解的原始 deadline 为准，gap 达到 1% 则结束。

上界初解在连续根前执行。此后每过约 45 秒主流程时间，若上界额度尚余、全局剩余大于 20 秒，就预约一次最多 6 秒的 ALNS repair。初解额度已耗尽但仍是 all-courier 时，预约优先选择新簇激活，不能因 `served=0` 跳过。`pending_design` 存在也不能永久挡住调度：当前整数 CG 在一轮 RMP/定价/加列的安全边界保存列、候选和已认证信息，返回 `yield_for_primal`，上界调用后继续该设计。

实现的第一步可给整数 CG 每个工作段设置不超过 30 秒的局部 deadline，并复用已有列与已验证点；重建成本必须计入，不能隐藏。后续如测得重建明显昂贵，再保留完整 native workspace 跨工作段恢复。中断未完成定价时不得宣称闭合；已生成的安全界/列/可行点继续保留。Benders 和 repair 的模型对象独立，临时固定和探索约束不进入全局主问题。

全局最后 10 秒不启动新 repair；用于已在手点的验证、最好解持久化和现有主流程安全结束。单次调用先预留至少 0.5 秒或该调用额度的 10% 作验证/保存，并结合最近实际验证成本调整；超过预算时返回既有已验证 best，不能回退到更差点。大文件导出在算法停止后的时间单列报告，不把其开销藏进或挪出别的模块时间。

上界新解反馈到 Benders：更新全局 UB/cutoff 与 master MIP start；合并实际产生的物理列；可把新设计加入候选队列。**不传递局部 MIP best bound，也不把其目标值当 Benders cut。** 同一设计的完整域固定设计 recourse 经现有认证接口验证后，才允许贡献普通 cut。

**10. 最佳解保存的强约束**

所有来源的可行点使用本轮修复后的统一提交通道。顺序为：原约束和完整成本验证 → 更新内存 best → 原子写入 checkpoint/身份 → 可选 cleanup/精修 → 验证更好版本后再次提交。清理超时/异常时保留清理前已验证点，不等待下一次外层 `account` 才记录改进。

最终输出从统一 best 取值，并检查 `final UB ≤ best validated checkpoint UB + tolerance`。检查点和最终结果的物理路径、成本、实例/目录身份应一致；只补回同一模型域的点。任何较差可行解只可能替换 current，不能替换 best。恢复 checkpoint 是结果恢复，不是一轮新优化，要记录来源和原始获得时刻。

**11. 建议的模块与接口**

| 模块 | 已有基础 | 待实现职责/返回约定 |
|---|---|---|
| 支持库/资源关联索引 | `PathCollection`、现有 `primal_portfolio.cpp` 分支索引 | 一次构造并增量补列；返回覆盖和完整性元数据；不再每个邻域扫描全目录 |
| `PrimalSearchState`（建议新类型） | 已有可行设计 pool | current、best 引用、精英池、算子权重、随机状态、预算和失败履历 |
| `PrimalRepairRequest`（建议新类型） | `CoupledPrimalNeighborhood` | 完整释放闭包、列索引、强化/探索模式、要求改变的组/强制关闭资产、局部额度 |
| repair 求解 | `solve_restricted_heuristic` | 复用原模型构建/原验证；新增探索限制；区分“全局兜底点”和“满足本邻域的 candidate”，不得将兜底伪作新邻居 |
| 精英精修 | `solve_fixed_design_cg`；目前 `CgColumnSeeds` 仅接入自由根/face | 为固定设计入口补上实际列种子或复用现有 pool 通道；限时继续，可行 UB 与认证状态分别标记。不能假定现有固定设计接口已支持传入该缓存 |
| 主调度 | `solve_benders` 与现有 pending design | 初解前置、周期窗口、CG 安全让出/恢复、共享 deadline、向 master 提交有效 incumbent |
| 最佳解提交 | 本轮修复后的保存通道、原子 checkpoint | 所有模块同一单调 best；中断不能丢解，局部约束不得污染全局身份 |

接口实现位置可沿用 [primal portfolio](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/primal_portfolio.hpp)、[受限修复接口](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/heuristics.hpp) 与 [CG 接口](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/benders.hpp)。上表带“建议”的类型和功能目前不存在，不能仅启用旧 `primal_seed_portfolio`/`primal_coupled_neighborhood` 就宣称已实现本方案。

**12. 主流程伪代码**

```text
start global deadline; validate and atomically retain all-courier / supplied incumbent
build or attach bounded branch-support bank and resource incidence index
run up to 3 distinct complete-branch seed repairs under the 30 s startup budget
current ← best feasible startup state; initialize elite pool and operator weights

while global time remains and global certified gap > 1%:
    if a primal window is due and primal budget remains:
        preserve any pending CG state at a safe boundary
        choose an eligible operator and head-block scale
        form old/new asset closure and complete-branch candidate columns
        retain required positive incumbent support
        if mandatory closure exceeds feasible resource budget:
            log reason; shrink seed block or select another operator
            continue
        create intensification or forced-change exploration repair
        solve original restricted SOCP/MISOCP under local deadline
        for each returned candidate that passes original validation:
            submit immediately if globally improving; persist before cleanup
            update current by feasible acceptance rule; update elite/cache
        charge construction + solve + validation + bookkeeping to primal budget
        update operator reward and schedule; optionally queue one elite refinement
    else:
        advance the existing certified Benders/CG search under a bounded work slice
        submit validated primal points immediately; retain valid global certificates
        retain columns and pending design if the slice yields before CG closure

finish from the shared best; verify monotone best/checkpoint identity and export
```

**13. 分阶段实现及验收，避免大范围消融**

1. **构造与可行 repair。** 先完成分支库、联合资源簇和 all-courier 启动，去掉单 head 正收益与固定 6-locker 门槛；用有明确已知改进解的小例验证“两个 head 各自不盈利、共享资产后盈利”的联合构造，以及缺一正率分支会使共同 rho 退化的反例。不要对任意实例断言一定能产生优于 all-courier 的解。
2. **资源闭包与搜索。** 实现六类算子中先可用的前五类、组级探索限制、current/best 分离和安全的劣解接受；验证关闭共享资源时所有依赖 head 均进入修复域、容量 one-hot 完整释放、全量旧正流支持保留、临时约束不泄漏到 master。
3. **预算与自适应。** 接入周期窗口、pending CG 的安全让出和总预算、算子奖励与精英池；用可控截止测试证明验证/清理中断不会丢弃已找到的 best，未知状态不会进入 LB；权重可以变化但必须可按 seed 重放。
4. **单例验收。** 功能验证和独立审查后，只用同一 L2 H20 OD40 R03，比“最佳解保存与定价/加列修复后的固定基线”与“加上完整上界模块”，各 600 秒、gap 1%、1 线程、seed 43、串行、输入/源/二进制冻结。CPLEX 旧结果若作参考须标成历史参考；如重新比较则同预算重新运行。无需各算子消融矩阵，第一轮也不扩 H40。

记录 30/60/120/300/600 秒的最佳已验证 UB、认证 LB、gap、首次正服务改进时间、实际 repair 次数、非空/覆盖失败比例、验证成功率、构造/求解/验证耗时、算子收益、不同设计数及最好解来源。报表同时核对 final 与 checkpoint 最小 UB，不能再次只读最终文件。新模块的成功首先应表现为更早得到低成本可行解；最终 UB 是否更好、认证 LB 是否因资源让渡而退步，需要同时报告。

首个性能目标可以设为“在前 60 秒获得优于 all-courier 的有效正服务解，并在 600 秒内改善固定基线 UB”；这是待检验目标，不是保证。1324.604929 与 1316.533353 用作本例已验证比较标尺，不用于偷偷构造种子或共享 benchmark incumbent。后续若单例有收益再扩实例；一次单例运行不能证明普遍优越，也不必为了开始实现先做消融研究。
