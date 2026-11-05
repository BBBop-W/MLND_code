# accelerated-20261105 之后的执行开销审计

本文审阅 2026-11-05 当前工作树，基准为已实现的 `accelerated-20261105`。只读取源码，没有读取实验目录、运行求解器或测量生产实例；因此下述结果是**可由控制流确认的重复工作及待验证的优化提案**，不是已经测得的瓶颈或加速倍率。证据全部来自实现本身，不需要外部 API 性能假设。本文未修改生产代码。

已经存在的零率列删除、head 消元、事件图缓存、同设计 RMP 延续、平滑发现列、支持补全和紧凑路径记录不再作为新建议。这里考察它们启用后仍然发生的计算。

记 $P,E,S$ 为注册服务路径、空驶路径和初始段数量，$C$ 为当前 RMP 的活动路径列数，$Z$ 为某个解的严格非零路径数，$V,M_A$ 为变量数和线性矩阵非零元数，$N$ 为基地与货柜节点总数，$B$ 为完整模型输入及段支持的身份快照字节数，$\ell$ 为路径物理记录的平均大小。表内优先级是实施顺序建议，不是实测收益排名。

| 项目 | 当前源码可确定的成本 | 建议顺序 | 仍须测量什么 |
|---|---|---|---|
| 1. 合并同一候选的残差计算 | 同一成功 RMP 点被多次遍历全部矩阵 | 第一批 | 验证耗时占比、失败/重试频率 |
| 2. 复用一次域身份计算 | 同一个持久化 slice 内至少两次完整快照构造 | 第一批 | $B$、快照构造/比较时间 |
| 3. 活动列身份增量维护 | 每次恢复与归还重复构造所有活动列 JSON 键 | 第一批至第二批 | $C\ell$、新增列比例、分配量 |
| 4. 内部候选使用类型化数据，按需打包 | 密集解数组、全目录遍历、段记录重复转换 | 第二批 | 包装次数、留存比例、峰值内存 |
| 5. 分离不可变物理检查与逐解检查 | 同一路径反复物理重构，反复建立名称索引 | 第二批 | 正支持重合度、物理重构时间 |
| 6. 跨固定设计重用参数化 RMP | 设计改变即重建矩阵及原生模型 | 单独评估 | 构建/提取时间与重新优化时间的权衡 |
| 7. 为补全建立完整分支倒排索引 | shortlist 缺口仍从整个目录起点扫描 | 按扫描计数决定 | 缺口频率、扫描命中位置、索引内存 |

## 1. 对同一原始点只计算一次完整残差

**证据。** `feasibility_diagnostics()` 先调用 `point_residuals()`，再调用 `feasible_point()`，然后自己再次遍历变量、线性行和锥来找最坏容差比。这是三次矩阵遍历。见 [benders.cpp:172](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:172)、[linear_model.cpp:601](/Users/songtaowang/Documents/MLND_code/cpp/src/linear_model.cpp:601)。普通新列集的一次成功整数 RMP 迭代中，`record_primal()` 调用该诊断；重试循环检查一次可行性；后面又检查一次可行性并调用一次完整诊断。其路径可静态数出 **8 次顶层线性行遍历**，条件是首次结果可行且最优、没有修改点、尚未耗尽重试计数。该计数不是运行时加速比例；诊断内部还可能重扫某个最坏行。见 [benders.cpp:914](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:914)、[benders.cpp:961](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:961)、[benders.cpp:984](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:984)。

**成本与建议。** 每次完整检查为 $O(V+M_A+\#cones)$。将残差计算返回类型化 `PointAssessment`，由同一个结果供重试分支、接受判断和日志使用；JSON 仅序列化评估结果。一个行循环中可以同时累积现有 double 门控量和 long-double 报告量，避免为了报告再次读矩阵。成功点收益结构明确；失败点有提前退出，实际收益需要单独测量。

**正确性义务。** 不能直接用 `point_residuals()["passed"]` 替代 `feasible_point()`：前者的变量界限使用绝对 `2e-6`，后者使用 `2e-7*(1+abs(x))`；线性求和精度也不同。第一版应保持两个现有判定的数值语义，只合并遍历和复用结果。任何点修复、重新求解、增列、行 RHS/边界变化都必须使旧评估失效。回归需覆盖贴近两类门槛、抵消严重的行、锥边界和修复前后点。

**可以直接定位的复用边界。** “8 次”仅为上述条件下的控制流静态计数，不能据此删除八个检查。缓存键至少包含 `(point_revision, matrix_revision, validation_policy)`；调用次数相近、向量地址相同或维度相同均不是充分条件。

| 具体调用对 | 向量/矩阵是否相同 | 可复用范围 |
|---|---|---|
| `subproblem` 第 984 行 `feasible_point(f.matrix, solved.values)` → 第 986 行 `feasibility_diagnostics(f.matrix, solved.values)` 内第 177 行同函数 | 相邻调用间只写日志字段，点和矩阵未改，容差/求和/语义完全相同 | 直接把第一次 `feasible_point` 的结果传入诊断；无需再次做该布尔检查。诊断的其他报告量仍需计算，或由同次评估提供。 |
| 第 921 行 `record_primal` 的完整诊断 → 第 986 行完整诊断 | 仅在无 `solved=...`、无 polish/修复、无新 solve 且矩阵 revision 不变时相同 | 同一成功原始点可直接复用完整诊断对象；发生恢复尝试后按真正选中的点重新绑定，不靠“最优”状态猜测。 |
| `feasibility_diagnostics` 第 176 行 `point_residuals` → 第 177 行 `feasible_point` | 点与矩阵相同，但界限容差、求和精度和返回语义不同 | 可共用一个循环内计算出的两套统计量，不能共用单个 `passed`。 |
| 启发式第 666 行原始 solver 点的残差 → `validate_result` 第 243 行 reconstructed 点的残差 | 后者经过导出设计取整和 lifting 重建，不保证是同一向量 | 不直接复用；只有逐元素身份或严格的点 revision 证明后才能共用。 |
| 矩阵检查 → `physical_residuals` | 语义层不同：前者检查建模矩阵，后者依据原始输入独立重建物理量 | 保留独立核验；第 5 项仅允许缓存与解无关的物理路径核验。 |

上述最后两项的调用证据见 [heuristics.cpp:664](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:664)、[heuristics.cpp:240](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:240)，导出取整见 [output.cpp:261](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:261)。

## 2. 域身份在一次调用内只生成一次，再考虑有约束的版本化

**证据。** `pricing_domain_identity()` 把输入矩阵、固定任务率和每个初始段逐项写入字节字符串。[pricing.cpp:60](/Users/songtaowang/Documents/MLND_code/cpp/src/pricing.cpp:60)。持久化 `subproblem()` 在入口计算一次该身份；稍后创建带缓存的 `LabelPricingOracle`，构造器再次计算相同身份。[benders.cpp:773](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:773)、[benders.cpp:853](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:853)、[pricing.cpp:105](/Users/songtaowang/Documents/MLND_code/cpp/src/pricing.cpp:105)。即使两个缓存都命中，也支付构造和相等比较成本。

**成本与建议。** 每次构造和比较为 (O$B$)，中间字符串也占 (O$B$) 内存。低风险版本是在一次 subproblem 内计算一个确切身份对象，传给状态缓存和事件图缓存，消除确定的重复构造。进一步可以在 solve 入口建立不可变 `SolveDomain`，将物理输入、初始支持、固定任务率分别版本化；事件图只依赖物理部分，任务率变化不必重建不受影响的物理图。事件图身份必须继续覆盖影响可行性和资源扩展的全部字段。

**正确性义务。** 目前 `Input` 和 `PathCollection` 的数据成员可以直接修改，因此仅比较对象地址、路径数量或调用方自报版本号并不足够。[input.hpp:21](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/input.hpp:21)、[paths.hpp:63](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/paths.hpp:63)。跨调用常数时间检查须先封装所有修改入口或持有实际不可变快照；无法做到时保留完整比较。新导入、预处理结果改变、支持变化或模型选项变化必须建立新身份，不能用未验证的普通哈希相等代替数学域相等。

## 3. 不再为全部活动列反复构造 JSON 身份键

**证据。** 状态恢复时 `recourse_columns_unchanged()` 对所有活动列重新调用 `seed_path_key()`；归还时清空并重建全部键。[benders.cpp:87](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:87)、[benders.cpp:1412](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1412)。键通过包含路径拓扑、资源和成本的 `Json::array(...).dump()` 构造。[benders.cpp:1704](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1704)。`CgColumnSeeds::include()` 同样先验证已有键，再合并/排序列 ID，然后清空并重新序列化全部键，即使只增加少量路径。[benders.cpp:1887](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1887)、[benders.cpp:1899](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1899)。新加的 `state_cache_snapshot` 计时已包含归还成本；这项计时修复不消除序列化本身。

**成本与建议。** 每次验证或完整重建约为 (O$C\ell$)，种子合并另有排序成本。先将确定不变的列键存于一个拥有明确生命周期的注册目录，状态只保存 ID 与其不可变记录引用；新增列只构造一次身份。种子集合使用有序增量合并并保留旧键。若暂不改所有权接口，可先缓存一个原始类型化 payload 快照并逐字段比较，避免检查时再创建中间 JSON 字符串；这种过渡方案仍为 (O$C\ell$)，不能宣传为常数时间缓存。

**正确性义务。** 该优化依赖“注册后只追加、已有 ID 的完整 payload 不变”，不是仅依赖签名相同。成本、时长、访问次数、segment ID 或顺序改变都要失效；签名只记录路线身份，不能替代完整系数身份。外部文件导入和手工构造 fixture 仍需完整核验。应保留现有同 ID 修改成本/资源会被拒绝的测试，并新增“只追加一列而不重建旧键”的计数检查。

## 4. 将内部候选表示与公开 JSON 输出分开

**证据。** `solution_json()` 先创建完整零解（包含 $P,E,S,N^2$ 大小的数组），随后覆盖 flow/head 等字段，再调用 `package()`。[output.cpp:27](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:27)、[output.cpp:257](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:257)。`package()` 计算指标并遍历所有路径生成记录，每次还把完整 `_segments` 转成 JSON；紧凑模式减少保存的 `_missions` 数量，但上述全目录遍历仍存在。[output.cpp:84](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:84)、[output.cpp:176](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:176)。`synchronize_result_paths()` 即使目录大小没变，也复制 `_solution` 并重新 package。[output.cpp:311](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:311)。每个非分数 subproblem 先打包全 courier 基线；每个可行 RMP 点在比较是否改善前又完整打包。[benders.cpp:821](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:821)、[benders.cpp:995](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:995)。

**成本与建议。** 每次至少 $O(P+E+S+N^2)$ 初始化/读取，加上非零物理记录和段 payload 转换。内部保留类型化设计、操作变量、严格非零支持与验证结果；只在 incumbent 接受、候选池需要、checkpoint 或最终导出时生成公开 JSON。公开稠密数组与原始 ID 可继续保持兼容。先做更小的改动也有价值：避免先填零再覆盖、为可信内部结果记录 `(solution_revision, catalogue_revision)`，在二者都未变时跳过同步重打包；新增路径只追加零 flow 并增量更新必要见证。

**正确性义务。** 不能把所有“未改善全局 UB”的候选丢弃：ALNS 的可行较差邻居和启发式设计池也可能有用，当前实现明确保留这些用途。[heuristics.cpp:673](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:673)。外部 JSON 进入验证时不能因为数组长度相同就信任其物理记录；现有校验先于同步覆盖记录的顺序必须保留。所有严格非零值，包括负值及很小的数值，仍须进入校验，不能用稀疏化阈值藏掉违规。

## 5. 保留独立物理验证，但缓存其不可变部分

**证据。** `validate_result()` 依次核验记录、同步打包、重建 lifting、检查矩阵及独立物理可行性。[heuristics.cpp:232](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:232)。`lifted_point()` 每次遍历全部变量，重建字符串到索引的 `std::map`，并把多个 JSON 数组转换为向量。[heuristics.cpp:28](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:28)。`physical_residuals()` 每次扫描全部服务 flow，对每个非零路径重新调用 `build_initial_segment()` 和 `build_service_path()`，再比较物理记录。[heuristics.cpp:115](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:115)。启发式主候选和 solution-pool 候选均走这条验证链。[heuristics.cpp:664](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:664)、[heuristics.cpp:707](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:707)。

**成本与建议。** 单次名称索引构建为 $O(V\log V)$；物理部分为 $O(P+E+Z\ell)$，外加矩阵验证。将 lifting 所需变量索引保存在 `Formulation` 的类型化布局中，增列时只延长必要映射。把“这条注册路径确实由当前输入产生，资源系数一致”作为不可变物理检查，每个域版本/路径版本做一次；逐解仍独立计算需求、节点平衡、容量、机队、成本等聚合量。若内部点已有可靠非零支持，可按该支持累积而非先遍历全目录；外部密集数组必须先完整扫描一次以建立支持并检查有限性/符号。

**正确性义务。** 缓存的是路径物理系数的核验结果，不能缓存某个设计或流量解的可行性结论。设施关闭、任务率、flow、rho 或任何模型约束改变仍需逐解验证。输入物理参数、同 ID payload 或初始段改变必须使物理检查失效；第三方导入仍做完整首次检查。不得让“物理独立验证”退化为直接相信 solver 矩阵，原始路径构造器的核验仍是可信来源。

## 6. 对非 conditional 的固定设计，用 RHS 更新扩展 RMP 复用范围

**证据。** 当前持久状态仅复用相同设计；`old->design != design` 会销毁旧状态并重建 `Formulation` 和 `NativeSolver`。[benders.cpp:768](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:768)、[benders.cpp:778](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:778)、[benders.cpp:837](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:837)。对固定设计，`build_formulation()` 添加 `fix_j: x_j=design_j` 并把设计变量目标系数置零；在 `conditional_demand=false` 时，没有把固定设计代入任务系数的步骤。conditional 模式则确实依赖设计计算 pool rate 和概率，必须排除。[formulation.cpp:272](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:272)、[formulation.cpp:285](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:285)。原生构造器重新创建变量、线性/锥表达式并提取模型；代码已经有按行修改 RHS 的封装。[linear_model.cpp:107](/Users/songtaowang/Documents/MLND_code/cpp/src/linear_model.cpp:107)、[linear_model.cpp:174](/Users/songtaowang/Documents/MLND_code/cpp/src/linear_model.cpp:174)。

**成本与建议。** 当前设计切换在应用层至少重做 $O(V+M_A+\#cones)$ 模型构造，原生求解器提取另计。可先在当前 preprocessed、非 conditional、固定整数设计范围实现一个参数化 recourse workspace：保留相同域、选项与路径列的矩阵，更新 `design_fix_rows` 的 RHS，追加缺列，再重新优化。不同设计可共享一个列集合的并集；这是同一完整模型的不同 RMP，列多也可能增加求解成本，需要列数/内存上限和重建策略。相比多设计缓存，单 workspace 不会同时持有多份大型原生模型。

**“同矩阵”的具体含义。** 对两个非空固定设计指针，在输入、建模选项、路径列及顺序相同且 `conditional_demand=false` 时，两次构造的变量类型、上下界、目标系数、所有行系数和锥结构相同，只有 `design_fix_rows[j]` 的 RHS 从旧设计值变成新设计值；战略变量在这两次构造中都保留为连续变量并被固定行约束，目标系数都置零。因此可以更新记录矩阵和原生固定行，而不是重新生成约束。这里并未声称“整个含 RHS 的模型对象相同”。代码中固定设计的其他实际读取只出现在 conditional 的 pool/probability 分支；当前预处理模式直接拒绝 `conditional_demand=true`，而 conditional 本身又只允许非松弛的固定整数设计。[formulation.cpp:212](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:212)、[formulation.cpp:222](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:222)、[formulation.cpp:273](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:273)。第一版建议仅支持当前 preprocessed 固定整数 recourse，不顺带扩大历史 conditional 或 face/free-design 范围。

**正确性义务。** 必须同时修改记录矩阵与 NativeSolver RHS，更新当前 design 身份，丢弃先前点的可行性、旧原生对偶/证书和数值恢复状态，重新验证新设计的全 courier 可行起点。每次全域下界仍需当前 solve 的对偶及完整定价。conditional 系数、free-design/face 额外行、模型选项或支持变化一律重建。现有同设计切片保留数值恢复 profile 的行为是有意设计，不能随意删掉其失败恢复隔离；跨设计复用应有独立的参数/状态重置路径。

这里能确定的是减少应用层重建工作；不能由源码推断 CPLEX 复优化必然更快，更不能假定 QCP 内部因子或基会按希望的方式复用。需要分别测量模型构建、提取、优化及恢复次数，再决定是否默认启用。

## 7. 完整分支倒排索引服务于 shortlist 缺口补全

**证据。** `PrimalSupportBank` 已经增量扫描目录并为每个任务分支保存少量候选，因此不应建议再次实现同一个 shortlist 缓存。[primal_support.cpp:87](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:87)。剩余成本在 `PrimalSupportBank::close()`：当关闭资源后的 shortlist 不覆盖某些正率分支时，从 `service_paths[0]` 开始扫描整个注册目录，逐路径构造分支 key 并过滤关闭资源；每个出现缺口的新邻域都可重复这件事。[primal_support.cpp:310](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:310)、[primal_support.cpp:332](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:332)。短名单评分及过滤也多次从同一路径构造 bases/lockers/airlines 集合。[primal_support.cpp:148](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:148)、[primal_support.cpp:319](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:319)。

**成本与建议。** 单次补全最坏 $O(P\ell)$，发生频率由缺失分支和资源关闭决定。建立覆盖所有注册路径的 `(first_od, departure, second_od) → ordered IDs` 倒排索引，保留当前候选排名表。补全只遍历缺失分支的 ID 列表，成本为 $O(\sum_{b\in missing}|P_b|\ell)$，最坏仍为 $O(P\ell)$；索引额外占 $O(P)$ 个 ID。为路径缓存稀疏资源集合或适合小资源域的位集合，可以减少反复分配 `set` 的工作。索引按 append-only 目录增量维护，ID 顺序保留，以免无意改变第一见证的选择规则。

**正确性义务。** 排名短名单和完整倒排索引必须明确区分；不能以 shortlist 空缺证明物理无支持。索引有自己的扫描 watermark 和完成标志：只有所有相关分支索引已覆盖完整物理目录、且闭合资源检查完整结束，才能给出“没有可行见证”的结论。中断保持 unknown 状态并继承现有预算逻辑。资源闭合是逐邻域条件，不能缓存某条路径对所有邻域都可用；不存在空驶返程也不能删除可由别的服务路径平衡的跨基地见证。

## 建议的验证顺序

先做 1、2 的局部去重，并增加调用次数/遍历量计数：它们不要求改变数学模型和公开输出。3 的低风险增量键维护可随后进行；常数时间版本检查应等所有权约束具备后再做。4、5 可共同建立“类型化候选—验证—按需导出”的接口，但分别测试输出逐字段一致性与物理核验完整性。6 应单独消融；7 则由现有 `support_completion.catalogue_paths_scanned` 计数判断是否值得额外索引内存。[primal_support.cpp:385](/Users/songtaowang/Documents/MLND_code/cpp/src/primal_support.cpp:385)。

所有比较应同时报告总墙钟、各阶段耗时、调用计数、实际列/矩阵规模与证书完成状态；同样时间内得到更好的可行解和更早完成全域定价是不同结果。当前没有执行这些测量，故尚不能给出收益排序或倍率。后续小 fixture 可验证等价性，性能判断则需要独立输出路径和固定预算的受控测量。
