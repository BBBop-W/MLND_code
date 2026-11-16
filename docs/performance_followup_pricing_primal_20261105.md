# accelerated-20261105 之后的列生成与上界搜索协同

本文以 2026-11-05 当前工作树为准，只读源码并核查公开一手文献；没有读取实验输出、运行求解器或修改算法。下述优先级表示建议的验证顺序，**不表示已经测得的瓶颈或性能收益**。所有新增策略均为研究提案。完整物理目录上的流式精确定价由[总报告第 3 节](/Users/songtaowang/Documents/MLND_code/docs/performance_followup_20261105.md)单独讨论，本文关注在相同定价后端上如何选择列、传递信息及安排时间；重复构建和验证开销见[代码审计](/Users/songtaowang/Documents/MLND_code/docs/performance_followup_code_20261105.md)。

本分报告的“第一批/第二批”只在协同策略内部排序；与等价执行优化及完整目录定价相比的总体次序，以总报告为准。

## 1. 当前已实现的起点

| 机制 | 当前实际行为 | 下一步需要增加什么 |
|---|---|---|
| 多列与批量插入 | `cg_max_columns=512`；定价保留按检验数、类型和签名排序的有界候选堆；全域最小检验数与返回列数分开计算 | 基于任务覆盖与资源差异选择同一批列；根据后续使用情况调整批量 |
| 平滑发现列 | 原始价格与平滑中心按固定权重更新；默认权重为当前价格的 0.5，每隔 4 轮原价格定价；发现列重新用当前原始价格验算，无可接受列时立即精确定价 | 依据原价格接受率和经认证进展调整权重、中心和精确定价间隔 |
| ALNS 支撑 | 按运行成本、增量资产成本、负荷与返航基地保留候选；缺分支时扫描注册路径池；关闭资产时释放依赖任务；已有分组距离 | 把 CG 路径使用量、价格及来源显式纳入短名单，而非只依赖静态排名 |
| 资源邻域 | 已有激活、共享、关闭/替换、拥堵、基地平衡、精英差异六类算子；支持路径引出资源闭包 | 在选择任务之前，构造跨多个任务分支的共享资源替换模板 |
| 自适应与预算 | ALNS 算子权重已经按收益与耗时更新，块大小会变化；总上界搜索额度为总时间的 30%，按 startup/regular/elite 分区，regular 冷却为 45 秒 | 在保持总额度与证书保底的条件下，根据不同阶段的已验证进展重分配窗口 |

证据：[preset 配置](/Users/songtaowang/Documents/MLND_code/cpp/src/solver_presets.cpp:121)、[候选堆与全域最小值](/Users/songtaowang/Documents/MLND_code/cpp/src/pricing.cpp:196)、[平滑与原价格回验](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1119)、[短名单](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:134)、[算子选择](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:244)、[已有耗时奖励](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_alns.cpp:354)、[预算规则](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/primal_schedule.hpp:12)。这些机制不能再作为“尚未采用”的建议。

统一记当前原模型 RMP 的线性行价格为 \(\pi_t\)，物理路径列为 \(a_p\)，其原始检验数为 \(\bar c_p^t=c_p-a_p^\top\pi_t\)。这里的 RMP 是含原始锥约束的运行子问题；不能直接把文献中 LP 单纯形的基进入率或收敛定理搬到本实现。以下列选择只改变求解顺序或受限原始可行域；完整下界仍须遵守当前原生锥对偶、遗漏列修正和全域定价证书流程。

## 2. 方向 A：有负检验数门槛的分层多列选择与批量控制

**优先级：第一批。** 当前候选堆只比较标量检验数及确定性次序，不能保证同一批列覆盖不同任务、返航基地或 locker 组合。这里可确认的是选择规则，不是“生成列必然重复无用”。已有 `batch_columns` 优化插入方式，也没有回答应当插入哪些列。[pricing.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/pricing.cpp:203)

**具体方案。** 第一版只改精确定价完成后的候选选择，不改标签可行性和支配规则：

1. 必留当前最负的遗漏列；在其余负列中，先按 \((\text{head},\text{origin},\text{task branch},\text{return base})\) 分层，各层取少量代表。若层数超过批量，轮换层次，并保持一部分名额用于全局最负列。
2. 在检验数合格集合 \(\bar c_p^t\le-\varepsilon\) 内，用资源特征 \(v_p\) 的距离破除相近候选之间的集中。特征包括标准化的 locker 占用、航线使用及基地净流出；不能只比较路线字符串。可先保留一个大小不超过 \(qB_t\) 的候选缓冲，再在其中选择 \(B_t\) 列。该缓冲的物理构造成本也须计时。
3. 可设第二道质量门槛 \(\bar c_p^t\le-\eta|\bar c_{\min}^t|\)，\(0<\eta\le1\)，再最大化资源差异；门槛无足够候选时退回负列集合。最低检验数见证始终保留，不能让多样性规则导致有负列却返回空批。
4. 每一至两次后续 RMP 重优化记录新列正流比例、资源覆盖和构建/优化耗时。当前是 QCP，应使用“原始正流参与度”，不要把它称为单纯形基进入率。批量从几个离散档位选择；正流参与度低而模型处理时间高时缩小，遗漏负列多且新增列持续参与时增大。先验证静态分层选择，再单独加入控制器，避免同时改变两个因素。

2026 年模板定价预印本提出在负检验数可接受域中优先追随协调模板，为“最负列并非唯一列选择标准”提供了直接研究背景；其主要试验是广义指派问题。上述分层与资源距离是本项目提案，并非该论文算法的复现。[Marshall、Shah、Dey，2026](https://arxiv.org/html/2604.12070v1#S3)

**数学边界。** 全域最小检验数、遗漏空驶列数及分支修正必须来自原有完整证书遍历，不能由多样性样本估计。返回列均需通过当前原价格回验。改变候选筛选可能使标签支配删除了本来更有多样性的路径：第一版只承诺在保留下来的候选中多样化；若要求全物理域的资源多样性，须另证相应支配规则，不应直接扩大宣称。

**最小验证。** 在可完整枚举的小实例上比较旧/新选择的全域最小检验数、修正量与证书状态；构造“最负列集中于一个 head、另一 head 有较弱负列”的例子验证层次覆盖；检查没有负列时不会人为添加证书贡献。随后用相同预算比较重优化次数、正流参与度、路径列数及认证时间，不能仅报告每轮插入列数下降。

## 3. 方向 B：让平滑强度和完整定价频率服从可观察进展

**优先级：第一批，与 A 分开做消融。** 当前平滑中心每轮均按固定权重更新，完整原价格调用按固定间隔安排；发现列仍需原价格接受，且没有可接受列即回退。这个安全结构应保留。[benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1121)、[原价格过滤器](/Users/songtaowang/Documents/MLND_code/cpp/src/pricing.cpp:549)

**具体方案。** 用 \(\widetilde\pi_t=(1-\alpha_t)\widehat\pi_t+\alpha_t\pi_t\) 表示发现价格，约定 \(\alpha_t\) 越大越接近当前原价格；该约定必须写进日志，避免与文献相反的权重定义混淆。每轮记录：

- 原价格接受率 \(A_t=\#\{\text{发现且原价格为负的列}\}/\max(1,\#\{\text{发现列}\})\)；
- 在固定设计、固定模型域内，最近完整定价后经修正的有效 recourse 下界变化；
- 发现列下一次优化的正流参与度，以及距离上一次完整定价的轮数和实际秒数。

先用保守的有限状态规则：接受率持续低或完整定价证书长期未更新时，增大 \(\alpha_t\) 并缩短间隔；接受率稳定且 RMP 有可验证进展时，才允许加大平滑、延长间隔。为间隔设硬上限，并在本次 slice 剩余时间接近一次历史完整定价耗时时提前安排完整调用，避免刚增列就耗尽证书窗口。历史耗时只是预算预测，不是能完成定价的保证。

进一步将“每轮无条件移动中心”改成“有经认证进展才更新主中心，另保留短期平滑量”。中心的比较须限制在相同设计和同一行布局内；不同设计 recourse 值的大小不能直接决定哪个中心更好。不能把原生 RMP 对偶值、未修正的发现价格或暂未定价的 RMP 目标当成完整域下界。

Pessoa 等的期刊摘要明确讨论平滑/罚函数的参数自调整和随算法阶段变化的调整；本文未获得其全文，因此上述状态规则是自主设计提案，不声称实现了其具体更新公式。[Pessoa 等，2018](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0784)

**数学边界。** 只调整发现过程，继续保留 `discovery_only` 与完整原价格证书的隔离。固定权重改为自适应并不自动获得论文中 LP 方法的收敛保证。若以后进一步跳过任务分支，必须有覆盖该分支的有效最小检验数下界；“之前没找到列”不构成理由。定价过滤论文明确区分有效下界过滤和会失去精确性的启发式过滤，也要求缓存真正的最小检验数，而非任一负列的值。[Bulaich Mehamdi 等，2025](https://arxiv.org/html/2411.11338v1#S3)

**最小验证。** 对同一组原价格/候选轨迹先离线回放控制规则，覆盖接受率为零、持续发现、近时限及缓存恢复；再用小全目录实例检验只有完整调用能置证书标志。实验分别比较固定 \(\alpha\)/固定间隔、只自适应权重、只自适应间隔、二者同时自适应；核心指标是获得完整证书的时间与总时间，不能用较少完整调用本身证明改善。

## 4. 方向 C：建立带来源约束的 CG → ALNS 反馈

**优先级：第一批至第二批。** 当前 ALNS 选中列会显式合入 CG 全局复用池；CG 与 ALNS 也通过注册目录和改善 incumbent 间接共享路径，所以“当前完全没有协同”并不准确。缺少的是将 CG 的价格、流量和发现来源显式传给 ALNS 的短名单排序。`observe_validated()` 只接受改善自身 best 的点，也未将一般 recourse 点作为可用路径证据入口。[列合并](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:3136)、[短名单指标](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:145)、[外部解接收](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_alns.cpp:396)

**具体方案。** 每次 CG slice 返回一个有严格大小上限的“候选列观察表”：路径 ID、模型/设计身份、当前原价格下检验数、经验证原始点上的流量、所属任务分支和资源特征、生成轮次。它不包含可被误用的全局界。ALNS 每个分支的短名单取以下几类候选的并集：现有低运行成本与低新增资产见证、最近原价格表现好的路径、曾在有效原始点承载流量的路径、资源组合不同的少量补充路径。固定总名额时先保障每个正率任务分支的支持，再削减可选广度。

这比把 \(\bar c_p\) 作为单一分数更合适：对当前连续 RMP 有用的列未必能改善整数设施方案，当前为正的检验数也不代表该路径在另一设计下无用。因此，原价格严格为负是**向 CG 插入发现列的门槛**，却不应成为 ALNS 路径候选的必要条件；ALNS 最终由原受限 MIP 决定正流和设施改变。可以把最近两个不同设计来源分别保留，避免一个设计的资源价格垄断短名单。

信息要附带“只用于排序”的作用域。收到不同输入、路径身份或任务率时全部失效；不同设计价格仅作为异地搜索偏好，不作新设计的 reduced-cost fixing 或不可行证明。超龄信息逐步降低排名，但不能因此删除 incumbent 的严格正流支持。列生成原始启发式文献强调，价格协调能够给启发式提供较全局的信息，动态变量又需要专门处理；它不证明本项目的具体反馈排名一定有效。[Joncour 等，2010](https://www.sciencedirect.com/science/article/pii/S1571065310000892)

**最小验证。** 构造一个“静态成本排名较差、但能复用 incumbent 已开资源”的路径，检验反馈是否真正进入短名单；再构造价格来自另一设计和旧任务率的情形，检验来源隔离。用小实例比较原短名单、仅流量反馈、仅价格反馈和二者并集；报告额外列数、实际候选使用率、独立验证通过率及首次/最终 UB。每次全局 UB 改善必须由原模型独立验证；观察表本身没有证书资格。

## 5. 方向 D：先选择共享资源替换模板，再闭合任务分支

**优先级：第二批。** 当前共享算子用一个随机锚点的资源重叠给任务打分；关闭/拥堵算子选一个 locker，再释放相关任务。启动 beam 已考虑联合激活，regular 块会扩张，分组半径也已实现。新增方向应研究“哪些改变必须共同发生”，而非再次建议扩大 ALNS。[当前任务打分](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:250)、[已有启动联合激活](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:234)

**具体方案。** 建立一个规模受限的任务—资源关联超图：任务头为一侧，基地、locker、航线及返航失衡方向为另一侧；边来自 incumbent 正流路径和 C 中少量备选路径。每次先选一个小替换模板，例如“两个旧 locker 的任务共同迁至一个备选 locker 组合”，或“两组相反基地净流出的服务同时调整”。选块时考虑当前使用与备选使用的差集，之后才调用现有资源依赖闭包和正分支补齐。

模板的粗分数可以同时奖励可回收固定成本与备选路径覆盖，惩罚预计新增资产和资源紧张度。固定任务分支流量为 \(\nu_{rb}v_{rl}\)，整数设计下 \(v_{rl}=\alpha_{rl}\rho_l\)，不能把已指派任务视为始终按完整率执行。代理分数可用明确标记的暂定利用率 \(\widehat\rho_l\) 估计分支流量并累积 locker 占用；不能直接把“有候选路径”当成容量可行。少量候选模板分别进入现有受限原模型求解；这一步才决定是否具有可行正流、容量档和节点平衡。

可以先做两个可解释算子：**双资源关闭—替换**和**服务净流出互补交换**，不要立即引入训练模型。若模板需要的容量组改变超过现有半径，应在明确的模板预算内选择扩大局部半径或放弃模板，记录该决策；不能悄悄把配置半径从 2 改成任意值。非探索模式继续包含 incumbent，强制关闭模式仍保留全局 fallback。

模板定价预印本使用来自当前原始解的协调模板来影响不同子问题的候选，这提供了“跨子问题协调”的启发。本文的路径流、容量与锥排队约束不同于其集合划分/GAP 模型，所以上述超图算子属于本项目新的启发式假设；不能沿用其整数兼容性或性能结论。[Marshall 等，2026](https://arxiv.org/html/2604.12070v1#S3)

**最小验证。** 用三个结构性反例检验设计：单独迁移不节省、联合迁移才节省固定成本；单向跨基地服务不平衡、两个方向联合才平衡；看似共享 locker 但容量或原锥约束不允许。比较普通共享算子与模板算子，保持总壁钟额度和候选 MIP 尺度尽量可比；分别报告闭包后任务数/locker 数、可行候选比例和经验证 UB，不将代理分数作为改进值。

## 6. 方向 E：按全局经验证进展调整搜索窗口，而非再次重写算子奖励

**优先级：第二批，先记录后控制。** ALNS 内部已有耗时折扣奖励，重新建议“按收益/秒调算子”会重复现有机制。真正尚未自适应的是 startup/regular/elite 的分区额度和窗口间隔，以及上界窗口与继续认证当前 recourse 的取舍。当前 regular 为最多 6 秒、elite 为最多 15 秒，非 startup 调用共享 45 秒冷却；这些是规则而非进展反馈。[PrimalSchedule](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/primal_schedule.hpp:19)、[窗口调用与全调用计时](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:3141)

**具体方案。** 第一版仍保留总 primal 额度 30% 和全调用计时，只允许把已确认无法使用的 startup/elite 额度按规则转移给 regular，或在额度内部调整窗口长度/间隔。为每类动作保留最近若干次的两类收益：经独立验证的 \(U\) 下降、经完整证书和主问题认可的 \(L\) 上升。可记录描述性效率

\[
e_a=\frac{w_U[U_{\rm before}-U_{\rm after}]_+
+w_L[L_{\rm after}-L_{\rm before}]_+}
{\max(1,|U_{\rm before}|)\,\Delta t_a}.
\]

这只是调度信号；权重需要事先定义，不能把两个标量加权后声称新的数学界。对 CG，RMP 目标改善、负列数量和新 cut 数量只能是中间进展，不能替代 \(L\) 改善；cut 的效果常在下一次 master 才显现，应单独记录延迟归因，避免把收益全部记给最后执行的动作。对 ALNS，较差可行邻居和新资源组合可作为探索信号，但不能记成 UB 改善。

需要保留最低探索次数、最大等待时间和当前 pending CG 的认证保底；否则短期奖励为零的动作可能被永久饿死。前期“尚无正服务 incumbent”、中期“上下界均有推进”、后期“仅剩证明工作”可以用已观察状态区分，分别设置较保守的规则，不必直接假设平稳多臂老虎机奖励。末尾保留的时间只能尽力支持最终验证与一次完整定价，无法承诺按时取得证书。

**最小验证。** 先在人工事件轨迹上验证所有阶段计费、失败调用、预算回收、延迟收益、无可行邻居和连续 pending 的行为。再比较固定规则、仅分区额度回收、仅间隔调整三组；固定总时限/线程/种子集，报告 \(U(t),L(t)\)、有效 gap 的时间积分、首次正服务时间、完整定价次数和认证间隔。过期、未定价完成或未验证的值不得进入这些曲线。本方向为本项目调度提案，不引用通用学习算法作为现成性能保证。

## 7. 已核实的一手文献及阅读深度

| 文献 | 已核实书目信息与入口 | 本次阅读深度 | 本文使用范围 |
|---|---|---|---|
| Pessoa 等 | Artur Pessoa, Ruslan Sadykov, Eduardo Uchoa, François Vanderbeck (2018), *Automation and Combination of Linear-Programming Based Stabilization Techniques in Column Generation*, INFORMS Journal on Computing 30(2):339–360. [出版社/DOI](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2017.0784) | 期刊元数据及摘要；HAL 版本无法抓取，没有声称阅读全文 | 只支持“参数自调整与阶段适应已被研究”；本文控制规则自行提出 |
| Joncour 等 | Cédric Joncour, Sophie Michel, Ruslan Sadykov, Dimitri Sverdlov, François Vanderbeck (2010), *Column Generation based Primal Heuristics*, Electronic Notes in Discrete Mathematics 36:695–702. [出版社/DOI](https://www.sciencedirect.com/science/article/pii/S1571065310000892)、[作者稿](https://www.math.u-bordeaux.fr/~rsadykov/papers/Joncour_etall_ISCO10.pdf) | 出版社摘要、作者稿索引中的题名/作者/摘要；PDF 正文抓取超时 | 支持价格协调与动态列环境下原始启发式的研究背景；不复述未读算法细节 |
| Bulaich Mehamdi 等 | Abdellah Bulaich Mehamdi, Mathieu Lacroix, Sébastien Martin (2025), *Pricing filtering in Dantzig-Wolfe decomposition*, Operations Research Letters 58:107207. [出版社/DOI](https://www.sciencedirect.com/science/article/pii/S0167637724001433)、[作者预印本](https://arxiv.org/html/2411.11338v1) | 出版元数据；2024 作者预印本正文 §3.1 命题 1、算法 1、§3.2 启发式过滤边界 | 用于区分可证明跳过定价与纯启发式跳过，不将前者误当当前已实现功能 |
| Marshall 等 | Luke Marshall, Prachi Shah, Santanu S. Dey (2026), *Accelerating Column Generation in Highly Degenerate Integer Programming Problems with Template Pricing*, arXiv:2604.12070v1. [作者预印本](https://arxiv.org/html/2604.12070v1)、[Microsoft Research 作者机构页](https://www.microsoft.com/en-us/research/publication/accelerating-column-generation-in-highly-degenerate-integer-programming-problems-with-template-pricing/?lang=ko-kr) | HTML 正文 §2.2–§3.2，尤其负检验数门槛、模板目标和当前原始解投影；未完整审阅所有证明与实验 | 提供协调列选择的研究启发；按预印本处理，没有确认正式发表，也不引用其跨问题加速倍率 |

检索日期：2026-11-05。所有文献均回到出版社、作者机构或作者预印本核实；全文未取得的条目已明确标记。没有向外部检索工具提交本项目源码、实例或实验数据。

## 8. 建议的最小研究顺序

先增加观察量而不改变策略：每批列原价格接受率、后续正流参与度、分支/资源覆盖、完整证书间隔、短名单候选的来源与最终使用情况。只读源码不足以判断 A、B、C 哪个收益最大。

第一轮分别做 **A 的静态多样性选择、B 的单一自适应因素、C 的有界候选反馈**；每项对照同一个 `accelerated-20261105`，保留其余设置。只有这些观察显示跨任务资源配合确有遗漏，才优先做 D。E 先用事件回放验证计费和防饥饿，再做受限额度调整。即使单项有收益，组合仍需重新验证，因为更大列批可能改变平滑接受率，而更强短名单又可能改变最佳 ALNS 时间份额。
