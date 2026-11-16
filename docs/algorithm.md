# 当前 Benders-CG 与加速策略

**C++ 当前实现（2026-10-19）**：已改为支持预处理、贪心初始列和动态路径池，不再依赖完整服务路径目录。规范流程、公共矩阵界、可行性和证书要求见 [隐式 CG 框架](implicit_cg.md)。新增的解析成本与资源割、最小割容量成本割、根节点割及联合优化启发式见 [主问题加强](master_strengthening.md)。下文七项加速和 2026-10-16 结果主要记录 Python 参考实现；其中依赖完整目录的成本强化不直接用于隐式 C++ 版本。

截至 2026-10-16，当前 `consecutive` 实现已接入下表的七个独立加速开关，包括适配当前 lifted SOCP 对偶的 Pareto 割。这里的“已实现”指代码接入和可分别启停，不等于每项都已证明提速。本轮已用冻结 TSL 20H 完成同版代码、逐例顺序执行的 120 秒消融：K1 开启 Pareto 后接受 103 条割，gap 从 5.446% 降至 5.270%，可行成本不变；K2 三次辅助求解均超时，接受零条割，开启/关闭的可行成本和下界相同。当前主要改善来自需求递推精简和主问题成本界强化，K2 的 Pareto 加速效果尚未建立。

当前版本六例的可行成本、下界、服务量、实际 wall、证书状态、源码哈希与收敛图见 [120 秒试验报告](../outputs/tsl_20h_algorithm_20261016/report/REPORT.md)。只有 K1 的完整路径参照达到 0.5% gap；四次 Benders 与 K2 完整路径参照均限时结束。不同阶段的结果和此前历史 `full_path` 结果分开记录，不能把阶段改动的效果全部归因于 Pareto。

## 基础算法与可选加速

令设计向量 `d` 包含设施启用、柜子容量档、OD 到出发/返回基地对的分配及航段启用。固定 `d` 后，`Q(d)` 是连续运营问题的最优费用，包含机队、服务路径、空调位及直接骑手配送费用。主问题最小化设施等固定费用与运营费用代理变量 `theta` 之和，普通 Benders 割用全局有效的仿射函数从下方约束 `theta`。

`full_path` 一次性求解完整有限路径 MISOCP；`benders_cg` 与 `benders_cg_plain` 使用相同的数学模型、路径目录、固定设计 SOCP 和普通全局割。普通割、SOCP 资源约束、完整目录定价、可行解独立验证及全骑手后备方案属于基础算法，并不计作下表的可选加速。

| 开关 | `benders_cg` 默认 | `benders_cg_plain` 默认 | 当前实际行为与边界 |
|---|---|---|---|
| `pareto_cuts` | 开 | 关 | 在普通割认证完成后求解核心点辅助对偶；通过完整定价和数值验收后额外加入 Pareto 风格割，保留普通割。 |
| `reuse_columns` | 开 | 关 | 跨固定设计 SOCP 复用已进入受限问题的服务/空调位列索引；不复用旧问题的对偶证书。 |
| `courier_savings_bound` | 开 | 关 | 用完整路径目录加入按分配计的最低服务成本界，以及按柜总占用计的最大节约界；无可行路径的分配不再获得免费服务节约。 |
| `initial_design` | 开 | 关 | 根据可行单单路径的需求加权运营节约贪心分配 OD；同池检查 K2 有向双单支持，包括 AA。提案须解固定设计 SOCP 并核验才能更新上界。 |
| `warm_start` | 开 | 关 | 每次求主问题前，将当前已验证 incumbent 的二元设计及运营成本设为主问题 MIP Start；未实现 SOCP 原始/对偶起点或基状态传递。 |
| `incumbent_cleanup` | 开 | 关 | 在保持所有路径流、分配、ρ、空闲库存和机队不变时，删除零流航段，选择支持实际柜负荷的最低费用档，并安全关闭闲置设施；前后独立核验。 |
| `primal_search` | 开 | 关 | 每五个外迭代，对变化后的 incumbent 最多尝试一次兼容的单 OD 增补设计，再解 SOCP；不是全面局部搜索、交换搜索或大邻域搜索。 |

上述默认值以 `src/solvers/methods.py` 为准，可用 `python main.py methods` 查看。CLI 允许重复使用 `--enable-acceleration NAME` 或 `--disable-acceleration NAME`，同一个开关不能同时启用和关闭。`full_path` 不接受 Benders 加速覆盖。

初始设计按 `demand × (ground_cost - path.cost - fleet_cost × path.time)` 排序；跨基地方案还扣除一条可行反向空调位任务的运营及忙碌资源成本。该分数只是启发式排序，不含精确空闲机队和设施固定费用。提案先打开全部候选设施、最大柜档及允许航段，让 SOCP 选流，再由 cleanup 删除冗余资源。初始 SOCP 最多使用 30 秒且不超过当时剩余预算的 15%；单次增补 SOCP 最多 10 秒且不超过剩余预算的 10%。它们没有独立于总预算的额外运行额度。

## 初始主问题成本界的两项强化

`courier_savings_bound` 现已替换此前“分配后可节省全部骑手费用”的粗界；CLI/API 开关名称不变，是否使用 Pareto 独立控制。以下两条约束直接由完整物理路径目录计算，不额外调用优化器，也不对 OD 分配强加必须正流的条件。

记全骑手成本 `C0=sum_r h_r*c_r`；对服务路径 p，`A_p=path.cost+fleet_cost*path.time`，`n_pr` 是服务 OD r 的订单次数，`k_p=sum_r n_pr`。完整忙碌时间中的飞行、等待和返回已计入 A；丢弃非负空闲库存与空调位成本只会降低运营成本下界。

第一条按 (r,b) 计算 `m_rb=min_{p含r且基地对为b}(A_p/k_p)`，`g_rb=max(c_r-m_rb,0)`；没有支持路径时直接设 g 为零。每条路径满足 `sum_r n_pr*m_rb<=A_p`，实际服务量又不超过 `h_r*a_rb`，故 `theta>=C0-sum_rb h_r*g_rb*a_rb` 全局有效。AA 的次数为二，不能按一个 OD 类别分摊成本；有亏损单的盈利双单组合也被包含。这条约束不要求每个已分配 OD 真正被服务。

第二条令净节约 `B_p=sum_r n_pr*c_r-A_p`，总柜占用 `T_p=sum_i slot_times_ip`。若没有正节约且零占用路径，可取 `beta=max_{p:T_p>0}max(B_p,0)/T_p`，空集合取零。由逐路径 `B_p<=beta*T_p` 及柜容量约束，得到 `theta>=C0-beta*sum_is threshold_is*z_is`；全部柜关闭时该界回到全骑手成本。slot_times 使用当前模型的预约与离柜占用系数，不自行把返回飞行再计入柜占用。若存在 `T_p=0,B_p>0` 的路径，停用第二条，保留第一条。

两条约束分别加入，不相加。它们同样适用于当前连续松弛，因为证明只需 OD 流守恒、`v<=assignment`、机队和柜负荷约束；不依赖分配变量的二元性。实现对系数作保守浮点方向修正，并将右端常数下移 `1e-8*(1+abs(C0))`。`master_bound_metadata` 记录是否启用、完整目录规模、未支持 OD、统一运营下界、实际 beta、停用原因和数值安全量。

冻结 TSL 的只读枚举系数记录位于 `outputs/tsl_20h_algorithm_20261016/master_bound_metrics.json`，该文件未调用真实算例求解器，不能作为算法收敛性能结果。有效性回归覆盖 K1/K2 微型网络全部 626 个合法固定设计，以及加速开关开/关与完整模型最优值一致性。此前只有粗骑手成本界的 pooled 消融属于上一源码版本；新旧结果必须注明实现差别，不把差异全部归因于 Pareto。

## Python 有限目录基准与 C++ 标号 CG

Python 方法保留完整有限目录扫描定价，作为参考实现。当前冻结 TSL 算例的路径域为直接飞行，覆盖当前关闭设施、关闭航段及未分配 OD 对应的路径；不能只在当前已开网络中找路后为其他设计生成割。

C++ `benders_cg` 从 2026-10-18 起使用 `cpp/src/pricing.cpp` 中的精确服务阶段标号定价。标号从输入网络逐阶段扩展，按有序出发—返回基地对处理单单、双单和空载任务；首单 OD、首段、忙碌时间、第二单自身时间、累计约化成本均保留。第二次取件处采用完整后续扩展支配：被支配标号的每个可行遗漏后续完成，都必须被另一个标号在相同交付点以不高于它的约化成本完成。比较包括已加入列的排除以及缺失列上界对偶项，不能仅比较当前位置和时间。

C++ 只在预处理中构造完整首段及分支行支持；完整任务路径由贪心和定价动态登记。航段界与初始成本/资源割均由输入数据推导，输出只保存已登记列。标号搜索不扫描目录中的完整路径来定价。它是当前 `K<=2`、直达路径域内的精确定价，不是任意 K、多跳路径或完整 branch-price-and-cut 实现。文献映射、支配证明与运行契约见 [label_pricing_research.md](label_pricing_research.md) 和 [C++ 说明](../cpp/README.md)。

固定设计的受限主问题保留完整首段及相应任务分支约束，仅省略部分完整路径的标量流变量。C++ 每轮先验证受限 SOCP 的原始最优解与显式对偶最优解及间隙，再执行完整标号定价，增量加入一批负约化成本列并重新求解。批量上限只限制加列数量，不限制精确搜索。无负列后还检查修正后的完整问题对偶间隙；超时、非最优对偶、定价未完成均不能产生新割。空列池有经过矩阵验证的全直接配送可行点，因此无需人工变量 Phase I。

同一固定设计内部的 CG 采用增量加列：Python 的 `add_path_columns` 使用 `gurobipy.Column`，C++ 的 `add_columns` 使用 Concert 增量变量及列系数。它们均沿用同一套约束行和物理列系数。新设计仍建立新的 SOCP；显式对偶仍按当前受限矩阵重新建立。

增量加列不随七个可选加速开关关闭。C++ `--plain` 同样保留精确标号定价及认证条件。当前已实现受限列 MISOCP 和可选 local branching；尚未实现 level/trust-region 稳定化或跨设计 SOCP 持久化。

## Aggregate 分组提升与 K=1 精简

同一基地对 `b=(l,m)`、同一首单 OD `r0` 的所有首段组成一组 `g`。在 aggregate 策略下，它们共享同一组分配变量，因此本轮保留每条首段的 `head_j` 及其与实际起始路径流的等式，仅将 eta、u、Z、Y 递推变量改为每组一套，并增加 `H_g=sum_j head_j`。分组来自完整首段目录，不随当前开放设计或受限路径列集改变。prefix 策略在正窗口时仍保留逐首段递推。

固定一个合法二元设计，定义 `Lambda_b=sum_r h_r*a[r,b]`、`q_b=1-exp(-delta*Lambda_b)`。原递推精确给出 `Z[j,t]=head_j*exp(-delta*sum_{r<t}h_r*a[r,b])`，以及 `u[j,r]=a[r,b]*eta_j`、`Y[j,r]=a[r,b]*Z[j,r]`。因此当 `Lambda_b>0` 时，每个首段都有 `eta_j=head_j*q_b/Lambda_b`，可以对同组递推直接求和。这里的 Lambda 和递推必须包含所有已分配 OD，包括 κ 为零的 OD；κ 只决定单单/双单任务分解，不决定到达池的总速率。

组变量仍具有有限且与设计无关的界。订单守恒给出 `H_g<=h_r0*a[r0,b]<=h_r0`。如果整数设计下 `H_g>0`，必有 `a[r0,b]=1`，所以 `Lambda_b>=h_r0`，从而 `eta_g=H_g*q_b/Lambda_b<=1`，同时 `eta_g<=delta*H_g`。这证明原来的 `min(1,delta*h_r0)` 上界可继续使用；不能仅凭逐首段 eta 的上界直接假设它们的和也不超过一。当 `Lambda_b=0` 时，订单守恒迫使组 head 全为零，eta 也为零。

反向恢复保持每条物理路径和 `head_j` 不变。当组内实际 head 之和严格大于零时，以 `head_j/sum_k head_k` 将组 eta/u/Z/Y 分摊回原首段；分母严格为零时恢复为零，不用正容差抹除很小的正流量。保存结果仍输出原首段长度的 eta，继续调用原有物理校验器。κ 只依赖首单和后续 OD，同组 single/double 分支和订单恒等式在上述双向映射下保持不变。

上述变换对整数设计的可行集合投影及固定设计 SOCP 最优成本精确等价。它们的分数主问题松弛不必相同：新的可行分数组解也能按比例拆回旧 McCormick 约束，例如 `t*(eta_g-U*(1-a))>=eta_j-U*(1-a)`；反向求和未必保持组 McCormick 界，因此新松弛可能更强，不会更弱。新矩阵和界仍仅由数据及完整目录决定，固定设计只改变固定行右端。对新公共连续扩展生成的普通割和 Pareto 割，对所有原整数策略仍全局有效。

`effective_delta=0`（包括 K=1）时 eta 恒为零，可完全省去 u/Z/Y 和双单分支行，并在输出时恢复零 eta。此精简与分组提升由三种方法共同使用，不计为可独立关闭的七项加速。它们改变了本轮完整路径模型的连续松弛和规模，因此旧版 `full_path` 结果可作为历史可行参照，但不能当作同源码性能对照。

验证包括 K=1/K=2 × aggregate/prefix 的完整 MISOCP及多个固定设计与原逐首段模型对照、原物理校验、不同固定设计矩阵不变、普通/Pareto 割跨设计有效性、κ=0 的到达池处理及极小正 head 恢复。独立原型和 88 次模型求解对照证据位于 `outputs/tsl_20h_algorithm_20261016/prototype/`。

## Pareto 割具体做了什么

普通 Benders 对偶可能有多个可行且在当前设计处同样紧的解，它们给出的割在其他设计处强弱不同。当前实现用 Magnanti–Wong 风格选择：保留当前设计附近的最优面，在设计空间内部的参考点上选择较强的支撑。

这里有三个不同的对象：

- **锚点 `d_anchor`**：当前已求过固定设计 SOCP 的二元设计。
- **锚点运营费用 `Q_anchor`**：该设计下已验证的连续运营解费用；只有普通割与它足够接近时才进入辅助步骤。
- **核心点 `d_core`**：用于比较割强弱的分数设计，不是一组实际部署方案。

核心点来自一个对所有合法二元设计都有正概率的分布：每个设施以 1/2 概率开启；已开柜点均匀选择容量档；端点均开启时，允许航段以 1/2 概率开启；每个 OD 在全部基地对和“不分配”之间均匀提出选择，仅在相应基地开启时接受。其期望在合法设计凸包的相对内部。完成一个外迭代后按 `0.50 × 旧核心点 + 0.49 × 当前设计 + 0.01 × 初始核心点` 更新，并检查设计等式和非固定不等式。

辅助问题仍采用当前公共 lifted 矩阵的显式锥对偶，包括所有变量有限上界/下界、锥项和设计固定行。设对偶解形成的仿射函数为 `L(d)`，辅助目标是最大化 `L(d_core)`，并要求 `L(d_anchor) >= Q_anchor - face_tolerance`。Python 基准一次补入全部遗漏路径的对偶平稳性约束及有限上界价格。C++ 则由精确标号定价逐批分离这些约束，重求辅助对偶，直至所有遗漏列均已认证；已分离列的上界价格必须计入目标和最优面条件。

普通割先保存。接受辅助割需要同时满足：

1. 普通割已有完整定价证书，且其锚点误差足够小；不紧的普通割不能冒充最优面入口。
2. 辅助求解器返回最优状态；辅助结果重新执行完整精确定价，不能沿用普通对偶的定价结论。
3. 对偶行符号、锥可行性、有限界价格及数值平稳性残差完成修正，得到全局有效的仿射下界。
4. 修正后的锚点仍在容差内紧贴 `Q_anchor`，核心点值不显著弱于普通割，且修正前后的核心点目标损失足够小。

默认 `face_tolerance = 1e-6 × (1 + |Q_anchor|)`，独立验证容差为 `2e-7 × (1 + |Q_anchor|)`；具体误差、核心点收益及状态均保存在迭代记录。辅助阶段最多使用 15 秒，同时受该次子问题预算的 20% 和实际剩余总预算约束。启发式初始设计/增补求解不调用 Pareto 辅助步骤。

辅助超时、非最优、定价未完成、锚点/核心点检查失败或辅助求解器异常时，保留已认证的普通割。通过验收后，主问题同时保留普通割和新割。严格的 Pareto 最优性需要精确最优面与精确辅助最优性；当前输出的 `certified` 表示上述数值条件下通过验证，不表示无误差的符号证明，也不保证实际运行更快。

## 输出指标怎样解释

每次报告必须同时给出可行成本 `objective`、有效下界 `bound`、`gap`、服务量、`termination_reason`、`pricing_complete`、`catalogue_complete`、实际耗时及输入/源码哈希。这里 `gap = max(0, objective - bound) / max(1, |objective|)`；`pricing_complete` 说明相应定价/割证书状态，并不表示整个整数问题已达到最优。

| 字段 | 定义 |
|---|---|
| `iterations` | 已完成并记录的普通 Benders 外迭代；其中包含该轮子问题、割及 Pareto 证据。 |
| `certified_cuts` | 加入主问题的普通割与已接受 Pareto 割总数。 |
| `master_bound_metadata` | 初始成本界的完整目录来源、实际 beta、保守数值下移量和统一运营下界；两条初始界不计入普通/Pareto 割数量。 |
| `acceleration_statistics.ordinary_cuts` | 普通有效割数量，包括初始设计或增补子问题产生的普通割。 |
| `pareto_cuts` / `pareto_attempts` | 已接受并加入的辅助割数量 / 尝试进入 Pareto 步骤的数量。 |
| `pareto_auxiliary_calls` / `pareto_auxiliary_solves` | 已调用辅助对偶的次数 / 返回求解器状态的次数；后者不等于辅助最优或接受次数。 |
| `pareto_rejections` | 按未接受原因累计次数，包括锚点不紧、辅助超时或检查失败。 |
| `ordinary_cg_iterations` / `heuristic_cg_iterations` | 普通外迭代子问题的 CG 次数 / 初始设计和增补子问题的 CG 次数；辅助对偶单独统计。 |
| `unique_service_columns` / `unique_empty_columns` | 所有已评估子问题中出现过的列索引并集大小，不重复计数。 |
| `reused_service_columns` / `reused_empty_columns` | 每次子问题传入复用池规模的累计值；同一列可跨轮重复计数，不能解释成独立生成的新列数量。 |
| `initial_design_attempts` / `primal_search_attempts` | 实际提交固定设计子问题的初始设计/增补次数。 |
| `primal_improvements` | 经核验后严格改善当前全局 incumbent 的次数。 |
| `cleanup_improvements` / `cleanup_savings` | cleanup 改善待考察可行解的次数/累计节约；待考察解可能仍劣于已有 incumbent，累计节约不等于最终总降本。 |
| `master_starts` | 给主问题设置 MIP Start 的次数，不表示求解器每次接受了该起点。 |
| `subproblem_timings` | 固定设计子问题的互斥阶段累计秒数：构模、原始求解、读解、独立核验、普通对偶、定价、割认证、加列、Pareto 和释放模型；包含启发式子问题，勿再叠加逐轮记录。 |
| `pareto_seconds` / `core_gain_total` | Pareto 总耗时 / 接受割在各自核心点的收益累计；核心点会变，收益累计不能解释成主问题下界提升。 |

API 的 `solver_wall` 包含路径枚举、求解流程和进度回调；运行器保存的 `wall` 还包括最终独立核验，不包括最终结果、CSV 和图表的导出。`enumeration_seconds` 保留旧口径（还含初始基线及起始进度），`enumeration_only_seconds` 仅计枚举调用。受预算停止时应保留验证通过的 incumbent 与已有有效下界，不能用受限路径模型的成本替代完整问题下界。七个开关的实际取值写入 `result.json` 的 `accelerations`；算法比较须检查实际值而不只看方法名称。

## 冻结 TSL 20H：严格回放与消融命令

固定输入位于 `data/benchmarks/tsl_2l20h50od`，规模为 2L/20H/50OD，总需求 2172.75 单/小时，κ 的 851 个可行配对已固定；仅在两个案例间改变 `max_orders=1/2`。完整参数、来源、哈希和历史参考见该目录的 `README.md`、`benchmark_manifest.json`。

以下命令在仓库根目录执行。严格回放使用 `--input`，**不同时传 `--config`**，因为完整物理配置覆盖会触发场景重建和 κ 重筛。显式指定各项求解预算，避免默认值变化。输出目录必须是新的或空目录。

```bash
python data/benchmarks/tsl_2l20h50od/verify.py
python main.py plan --input data/benchmarks/tsl_2l20h50od --method benders_cg --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100
```

完整加速版与仅关闭 Pareto 的成对比较，其余开关保持一致；每条 `compare` 顺序运行 K1 与 K2，每个 K 各有 600 秒：

```bash
python main.py compare --input data/benchmarks/tsl_2l20h50od --method benders_cg --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100 --out outputs/tsl_20h_algorithm/benders_cg
python main.py compare --input data/benchmarks/tsl_2l20h50od --method benders_cg --disable-acceleration pareto_cuts --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100 --out outputs/tsl_20h_algorithm/no_pareto
```

若要测量在无其他可选加速时单独添加 Pareto 的影响，使用下列另一组配对；不能把它与上一组的差异全部归因于 Pareto：

```bash
python main.py compare --input data/benchmarks/tsl_2l20h50od --method benders_cg_plain --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100 --out outputs/tsl_20h_algorithm/plain
python main.py compare --input data/benchmarks/tsl_2l20h50od --method benders_cg_plain --enable-acceleration pareto_cuts --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100 --out outputs/tsl_20h_algorithm/plain_plus_pareto
```

同版完整路径参照可用相同参数，将方法改为 `full_path`、输出目录改为 `outputs/tsl_20h_algorithm/full_path`，不传加速开关。此前保存的 20H `full_path` 结果属于历史参照，当前轮次的源码哈希不同；10H 结果不是当前固定实例的算法性能对照。本文不将历史结果或小型正确性测试视为本轮真实 20H 的加速效果结论。

实现位置：`src/solvers/{api,methods,algorithm,formulation,dual,pareto,heuristics,strengthening}.py`；针对性验证位于 `tests/unit/test_{solver,pareto,pooling,accelerations,heuristics,master_bounds}.py`。

## 分阶段计时口径

`acceleration_statistics.master_solver_timings.optimize` 为主问题所有 `model.optimize()` 调用的墙钟时间；即使最后一次求解直接触发停止、没有生成外迭代记录，也计入总数。`calls` 是实际进入优化器的次数。

`subproblem_timing_details.primal_solver.optimize` 为固定设计运营 SOCP 的优化器时间；`dual.optimize` 为普通 Benders 割的显式对偶优化器时间；`pareto.dual.optimize` 为 Pareto 辅助对偶优化器时间。普通和辅助对偶还各自拆分 `build / postprocess / dispose`。嵌套的 `solver_runtime` 是后端诊断，不能再与 `optimize` 相加。`subproblem_timings.dual / pareto` 是父阶段耗时，也不能与其子叶相加。

定价采用完整有限路径目录扫描，其耗时为 `subproblem_timings.pricing + subproblem_timing_details.pareto.pricing`。后项用于核验辅助对偶，已经包含在 Pareto 父阶段内。它没有单独的 MIP/SOCP 优化器调用。

`algorithm_timings` 记录基线准备、主问题构建/更新、启发式设计准备、候选解核验和清理、检查点、释放主模型。`incumbent_cleanup` 包含清理函数内部的两次核验。`runner_timings` 单列检查点核验、检查点写入、进度 I/O、最终核验；它的两个检查点项属于 `algorithm_timings.checkpoint` 的子项。统计总占比时必须选择父项或子项，不能重复相加。尚未被阶段计时包围的控制、转换和初始化工作保留为明确标注的余项，不能当作主问题求解时间。

2026-10-16 的计时回放和字段核对见 `outputs/tsl_20h_timing_20261016/`。该回放仅新增计时，保留冻结的 TSL 2L/20H/50OD 输入、全部七项加速和每例 120 秒预算；它不是旧结果的逐秒追溯。
