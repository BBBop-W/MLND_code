# consecutive：C++ / CPLEX 实现

此目录提供原生 C++17 求解入口，直接读取 `consecutive-csv/1` 的 CSV/JSON 实例，通过隐式列生成或全路径基准调用 CPLEX Concert，并写出 JSON/CSV 结果。**读取、求解、结果输出均不需要 Python 进程或 NPZ 文件**。Python 代码用于独立复核、Gurobi 对照和后续分析。

固定测试实例为 `examples/tsl_2l20h50od`：2 个基地、20 个柜候选点、50 个 OD。K=1、K=2 共用该目录，通过 `--k` 选择，所有地理、需求和成本系数保持相同。格式及参数定义见实例的 `README.md`。

`examples/` 中的实例采用原生 CSV/JSON 格式，供本目录程序或 `python -m src.experiments.native gurobi --instance ...` 使用。Python 常规入口 `main.py run --input` 需要 `input_metadata.json` / `input_arrays.npz` 快照，例如 `data/benchmarks/tsl_2l20h50od/`；两种目录不能互换。

当前分支的 **K=1 使用 OD × 出发基地分配**，返回基地由服务路径选择；K=2 仍使用 OD × 出发基地 × 返回基地分配。路径中的出发/返回端点、基地流量平衡和空飞调拨均保留。两种 K 的对照现在同时包含分配域差异；历史 K=1 基地对结果也不能直接当作当前 K=1 的同模型算法基准。公式和迁移边界见 [K=1 起点分配说明](../docs/k1_origin_assignment.md)。

K=1 输出 `assignment_scope="origin"`，分配及可用率乘积数组为 `(R,L)`；K=2 保留基地对数组。K=1 明确拒绝 `--assignment-support-cuts`、`--double-support-cuts`、`--joint-base-cost-cuts`：这三项当前使用固定返回基地的证书。一般成本下界在可选返回基地上取最小值；资源 LP 保留交付和出发侧门控，K=1 不使用返回分配门控。公共模型、完整路径、Benders 和 CG 仍保留完整返回路径。

## 代码组成与复用

| 文件组 | 职责 |
|---|---|
| `include/mlnd/` | 输入、路径、矩阵、求解结果和算法的公共类型与接口 |
| `src/input.cpp`、`main.cpp` | 标准 CSV/JSON 实例校验、配置、命令行和运行入口 |
| `src/paths.cpp` | 首段、完整服务/空飞路径及统一资源系数；完整枚举和定价共用 |
| `src/formulation.cpp`、`linear_model.cpp` | 设计与运营约束、稀疏列系数、锥模型及 Concert 调用 |
| `src/benders.cpp`、`dual_workspace.cpp` | 外层 Benders、固定设计 SOCP、认证对偶、列池和增量对偶 LP |
| `src/pricing.cpp` | 精确标号扩展、支配、约化成本与完整定价证书 |
| `src/heuristics.cpp` | 初始设计、列池 MIP、邻域搜索等可行上界策略 |
| `src/strengthening.cpp`、`topology_cuts.cpp`、`assignment_support_cuts.cpp`、`branch_joint_cuts.cpp` | 公共下界、路径支持及组合成本割；分别检查其适用分配域 |
| `src/bdd.cpp`、`bdd_search.cpp` | 独立 BDD 诊断及乘子搜索，与默认算法区分 |
| `src/output.cpp`、`progress.cpp` | 统一结果、运营表、来源记录及运行进度 |

修改模型时先改公共构造与索引，再核对完整模型、Benders、定价、固定设计和导出；不要分别维护几套需求方程。`cpp/tests/` 中的探针由 Makefile 的对应目标构建，用于矩阵、固定设计、割、定价和求解器互验。实际可用目标以 `cpp/Makefile` 为准。

原生结果的 Python 互验入口是 `src/experiments/native.py`；已验证结果的报告使用 `src/reporting/native.py` 或相应批次的 `scale_ladder.py`、`structural_comparison.py`、`bdd.py`。所有图形共用 `src/reporting/style.py`，不要另写一套 CPLEX 专用绘图样式。输入布局和现成命令见 [根 README 的报告复用指南](../README.md#报告与绘图优先复用已有入口)。

可选的 `--od-airline-bounds` 在 Benders 外部资源 LP 中加入按 OD 需求分解的航线门控，再投影为主问题成本割；不增加主问题变量。它适用于 lifted 和 conditional 两种需求表示，默认关闭。这些割只保证对整数设计及其凸包有效，开启后的 `root_diagnostics` 与原 `continuous_root` 分开报告，不能相互拼接计算连续 gap。数学证明、测试和对照协议见 [OD 航线割说明](../docs/od_airline_cuts_20261022.md)。

## 构建

从仓库根目录执行：

```sh
make -C cpp -j2
cpp/build/mlnd_cpp --help
```

需要 C++17 编译器、CPLEX Optimization Studio 的 C++ Concert 头文件/库，以及 `nlohmann/json.hpp`。本机默认配置是：

```text
CPLEX_ROOT       = $HOME/Applications/CPLEX_Studio222
NLOHMANN_INCLUDE = /opt/anaconda3/include
```

JSON 依赖只使用 C++ 头文件；路径包含 Anaconda 不代表运行时调用 Python。修改安装位置时：

```sh
make -C cpp -j2 \
  CPLEX_ROOT=/path/to/CPLEX_Studio222 \
  NLOHMANN_INCLUDE=/path/to/include
```

`NLOHMANN_INCLUDE` 指向包含 `nlohmann/` 子目录的目录。Makefile 自动选择本机 macOS arm64/x86-64 库路径，也允许覆盖 `CPLEX_PLATFORM`、`CXX`、`BUILD_DIR`。Linux 配置还需要 OpenSSL Crypto 开发库。跨平台构建配置已提供；本轮执行验证针对本机 macOS arm64。若本机已安装 CMake，也可使用 `cpp/CMakeLists.txt`，对应 JSON 配置变量名为 `NLOHMANN_JSON_INCLUDE_DIR`。

## 读取与完整模型

只检查输入，不运行优化：

```sh
cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 1 --inspect
```

读取器按 `model.json.ids` 对齐 CSV，检查全部输入文件的 SHA-256、单位、维度、数据类型、ID 唯一性、完整系数表和取值域，并核对观测数量与需求率、泊松容量阈值。来源记录中的历史绝对路径只用于追溯，读取器不会访问这些路径。

完整模型 K=1：

```sh
cpp/build/mlnd_cpp \
  --input examples/tsl_2l20h50od --k 1 \
  --method full_path --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/cpp_tsl20h_full_K1
```

完整模型 K=2：

```sh
cpp/build/mlnd_cpp \
  --input examples/tsl_2l20h50od --k 2 \
  --method full_path --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/cpp_tsl20h_full_K2
```

每次使用新的输出目录；程序拒绝覆盖已有 `result.json`。`--gap 0.005` 表示 0.5% 相对最优性间隙。省略 `--k` 时采用实例默认值，本例为 K=2。`--log` 打开 CPLEX 日志。

当前原生路径枚举器支持 `route_mode="direct"` 且 `candidate_routes=null`。其他路径域会被明确拒绝，不能把被拒绝的输入解释为自动采用直接路径。模型读取原有 `flow_policy`：`aggregate` 按同一基地对及第一单 OD 汇总分支比例，`prefix` 分别约束第一程；读取器不会在两者之间转换。固定 TSL 实例采用 `aggregate`，其等价验证也采用该规则。

`full_path` 将完整可行目录中的所有服务路径和空飞路径直接放入 MISOCP。`benders_full` 也使用完整目录；`benders_cg` 仅预处理首段/分支支持，以动态列池和精确标号定价求解，不预枚举完整路径。

## 固定设计和初始解

`--fixed-design` 固定 `facility`、`capacity_level`、`assignment`、`airline` 四组 0/1 决策，然后求解连续 SOCP；支持 `full_path`、`benders_full` 和 `benders_cg`，最后一种使用隐式列生成：

```sh
cpp/build/mlnd_cpp \
  --input examples/tsl_2l20h50od --k 2 --method full_path \
  --fixed-design outputs/cpp_tsl20h_full_K2/solution.json \
  --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/cpp_tsl20h_fixed_K2
```

`--start FILE` 提供初始设计：完整模型把四组设计变量作为部分 MIP start；Benders 先评估该设计。其他连续变量不会作为跨求解器热启动值原样使用。输入文件可以是原生 `solution.json`，或包含 `_solution` 对象的 JSON；原生入口不读取 Python 的 `.npz`。设计文件应来自同一个实例及相同 ID 顺序，分配维度必须匹配本次模型。当前 K=1 与 K=2 的分配域不同，不能直接跨 K 传入旧三维分配；如需利用旧设计，应显式转换、核验，并重新求解运营流量。

固定设计的求解器目标仅含运营成本；对外输出的 `objective` 和 `bound` 会加回固定设计的设施成本，保持与完整模型一致的总成本口径。

## Benders 入口与加速记录

```sh
cpp/build/mlnd_cpp \
  --input examples/tsl_2l20h50od --k 2 --method benders_cg \
  --seconds 600 --gap 0.005 --threads 1 --seed 43 --max-iterations 1000 \
  --out outputs/cpp_tsl20h_benders_cg_K2
```

将方法改为 `benders_full`，可使用包含全部路径的 Benders 子问题，关闭子问题列生成。`--max-iterations` 限制外层 Benders 迭代次数。`--no-pareto` 关闭 Pareto cut 辅助问题，`--no-reuse-columns` 关闭跨设计列池复用。

### 固定分配的条件 Benders（2026-10-22）

`--demand-formulation conditional` 是可选算法模式，默认仍为 `lifted`。
它适用于 `benders_cg`、`benders_full` 及其固定整数设计入口。
在整数运营子问题中按当前二元分配计算既有 `q*kappa*share` 系数，精确消去 `eta/u/Z/Y`，保留 `head/group_head/v`、所有单单/双单比例和订单守恒行及物理约束。
两种 `flow_policy` 的含义、K 和路径可行域不变；公共输出的 eta 根据每条首程的实际 head 恢复。

认证的对偶仿射界先吸收当前 alpha 系数，再按全部 alpha 位串添加 Hamming 条件项。
大 M 由仿射函数在其余二元设计的盒上界计算并向安全方向舍入，不用当前子问题值代替。
数值对偶修正、完整域定价、跨设计列复用、射线 LP、容量/资源割、受限 MIP、可行解清理和自适应主问题均复用现有实现。
Pareto 核心点投影到当前 alpha 的设计面，所有整数子问题的普通/辅助割均进行条件化。

连续根节点及分数设计子问题继续使用原 lifted 松弛；受限 MIP 也使用旧表达。
临时连续根主问题排除仅在整数分配域有效的条件割，使其连续上下界口径一致；这些割仍保留在整数主问题中。
完整 `full_path` 基准不接受 conditional 开关。

结果新增 `demand_formulation`、`demand_formulation_scope`、`master_model_size`、`subproblem_model_size` 及 `subproblem_model_size_peak`。
子问题规模只统计整数设计运营子问题（含启发式设计），不混入 lifted 连续根节点。
每条条件割记录 alpha 位串、M、额外舍入修正和锚点前后值；`acceleration_statistics` 记录条件割及子问题次数。

```sh
make -C cpp -j2 conditional-cut-probe conditional-formulation-probe conditional-benders-probe
cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 2 \
  --method benders_cg --demand-formulation conditional \
  --seconds 180 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/NEW_CONDITIONAL_K2
```

数学消元与条件割见 [实现手册](../docs/conditional_benders_handout_20261022.md)，既有标准算例四方法比较见 [实验说明](../docs/conditional_benchmark_20261022.md)。

### 精确标号列生成（2026-10-18）

`benders_cg` 的子问题是连续 SOCP；外层选址与分配仍由 Benders 主问题处理。这里迁移的是多基地路径算法的精确定价方法，未将整个求解器改称 branch-price-and-cut。文献依据及区别见 [定价研究](../docs/label_pricing_research.md)。Python 扫描定价与 `full_path` 保留为参考基准。

每个固定设计执行以下流程，`--plain` 也保留这些必要步骤：

```text
建立受限 SOCP：完整公共约束 + 少量贪心种子列（可行性不依赖种子）
验证全直接配送可行点：rho、任务流和无人机为 0，直接配送量等于需求
循环：
    求解受限 SOCP，检查最优状态及原始可行性
    求解其显式锥对偶，检查最优状态、对偶残差及原始—对偶间隙
    对全部有序出发—返回基地对进行精确标号定价
    若搜索未完成：保留已验证可行解，退出，不生成新割
    若有负约化成本列：加入至多 cg_max_columns 条，回到受限 SOCP
    否则：修正剩余浮点约化成本误差，并检查完整问题对偶间隙
          通过后输出子问题最优性证书和全局有效普通 Benders 割
```

空列池具有经过矩阵检查的可行点，因此本模型不需要人工变量 Phase I。不能把另一个选址的全关闭基准替代用户指定的固定设计。任务分支约束及 κ 不因列池缺少路径而删除或重算。

**标号和扩展。** `src/pricing.cpp` 按出发、首次取件、首次交付、第二次取件、第二次交付及返回的拓扑顺序扩展。状态保留基地对、首单及首段、第二单 OD、当前位置、任务忙碌时间、第二单自身取件时间和累计约化成本。负弧成本允许存在，因为服务阶段图无环，算法不依赖 Dijkstra 的非负边假设。AA、不同阶段重复访问柜子、跨基地返回均保留；空载调拨也定价。任务时间与各订单时限使用 `paths.cpp` 的同一权威资源构造器检查。

**支配证明。** 同一第二取件状态的候选标号按 `(出发基地, 返回基地, 第二单 OD, 第二取件柜)` 分桶。先比较时间和累计约化成本作为必要筛选，再比较剩余至多 H 种交付点的完整扩展。只有被删标号的每一个可行遗漏完成都能被保留标号在相同交付点以不高于它的完整约化成本替代，才能删除它。比较之前排除已加入 RMP 或已分离的路径，并计入逐路径上界对偶价格。因而不同历史可以被支配，同时不会把“较便宜路径已经加入，另一路径仍应加入”的情况误删。支配关系保持最小遗漏约化成本；它不承诺保留全部负列或全体路径中最便宜的前 B 条。

**约化成本与证书。** 普通遗漏列取 `rc_p = c_p - pi^T A_p`；Pareto 辅助对偶中已经补入的路径还需计入非负上界乘子。所有线性行对偶均来自当前 RMP；路径不直接进入锥约束，机队和可用率通过共同模型的线性行和锥对偶关联。记录的最小约化成本是 `min(0, min_p rc_p)`。若最终存在容差内微小负值，使用 `D*max(0,-min_rc_service)` 加上 `D*N_empty_omitted*max(0,-min_rc_empty)` 下移割的常数；D 为总订单需求。前项利用订单守恒推出的 `sum service_task_flow <= D`，后项利用每条空载路径的独立流量上界 D。修正后的原始—对偶间隙仍必须达标。

**对偶数值恢复。** 显式锥对偶若返回 `NumBest` 或修正后的间隙不合格，会在同一预算内尝试数值设置及 `cone_ray_lp_recovery`。对于原始锥 `w² <= F*u*v`，从当前原始最优解构造对偶可行方向 `(sqrt(F*v/u), sqrt(F*u/v), -2*sign(w))`，向可行侧修正舍入误差，再让非负倍率由 LP 优化。此 LP 将原对偶锥限制在一条可行射线上，因此其解始终可作为原对偶可行证书，不是放松原模型；锥紧时该方向对应互补方向，锥不紧时倍率 0 仍允许。LP 最优本身不证明原对偶最优，必须另经同一原始—对偶间隙及完整标号定价认证。`dual_attempts` 保留失败状态，`dual_certificate_method` 区分恢复方法；没有把 `NumBest` 改写为原锥对偶 `Optimal`。恢复仅用于普通 RMP 对偶，不据此宣称 Pareto 辅助问题最优。

Pareto 辅助对偶使用相同标号算法逐批分离遗漏路径驻点约束，重求辅助对偶后重新定价；上界乘子进入辅助目标和锚点最优面约束。辅助步骤失败时保留此前普通认证割。`accepted` 表示数值容差下通过验证的 Magnanti–Wong 风格加强割，不声称浮点求解产生零容差的理论 Pareto 证明。

`--cg-max-columns` 默认 64，只限制每轮加列量，不截断标号搜索；`--pricing-tolerance` 默认 `1e-7`，是绝对约化成本容差；`--cg-duality-tolerance` 默认 `1e-6`，乘以 `1+abs(运营成本)` 得到原始—对偶间隙容差。每轮 `cg_trace` 记录 RMP/对偶状态、目标、间隙、定价是否完整、最小约化成本、生成/扩展/支配标号数、加列数和各阶段耗时。`negative_paths`/`omitted_paths` 统计支配后被检查的完成路径，不是完整目录中所有负列/遗漏列的总数。

预处理只保存首段/任务分支支持；路径由贪心或定价生成时动态登记、去重并获得稳定 ID。先生成便宜的单单/双单种子，再在限定预算内补齐单单及每个正需求、正 κ 的续单分支。`aggregate` 分支可选不同首段；`prefix` 寻找共用一个首段的完整分支包。首选首段无可行延长时继续尝试其他首段。该步骤不固定最终选址或流量，物理缺失、预算中断和资源不足也不会导致分支等式被删除。定价的负列堆受批量上限限制，后缀缓存最多 128 组。框架解释见 [隐式 CG 框架](../docs/implicit_cg.md)，本次种子与主问题加强以 [主问题加强文档](../docs/master_strengthening.md) 为准。

### 主问题加强与可行上界搜索（2026-10-19）

主问题现在从输入航段与接驳成本推导每单的最低运营成本，加入分配节省割；同时用每单最低格口占用时间推导全域有效的柜容量节省割。所有极值只依赖输入，不能由当前已生成路径子集计算。主问题只保留设计二元变量和一个运营成本变量 θ。外部货币最大流网络在根节点及整数候选处生成容量成本割，新增行不增加变量。原主问题的 S、ρ、空闲无人机及其相关松弛已删除；完整运营子问题保持原模型。证明、浮点放松方向和零分母处理见 [主问题加强文档](../docs/master_strengthening.md)。

正式外层搜索前，默认先在外部通过 CG 求战略自由变化的连续松弛，将完整定价认证的下界作为一条 `F(x)+θ≥L_R` 总成本行写回。阶段预算为 `--continuous-root-seconds`（默认 120 秒）、`--continuous-root-budget-fraction`（默认 65%）及总剩余时间的最小值；`--disable continuous_root` 可关闭。该阶段不枚举完整路径、不增加主问题变量，分数解不进入整数上界。`continuous_root` 单独记录原始连续可行目标、认证下界、完整定价状态、残差及耗时。详见 [连续松弛 CG](../docs/continuous_root_cg.md)。

若上述连续上下界尚未闭合，再进行最多 20 轮旧根节点割生成，预算比例默认 10%。`root_diagnostics` 记录连续上下界、运营成本低估量和停止原因；轮数用尽不代表分离收敛。分数设计使用原共同运营矩阵的连续松弛，只有原始—对偶间隙和完整定价认证通过才加入根节点割。分数运营解不进入整数可行上界；命令行 `--fixed-design` 仍只接受二元设计。连续松弛已认证闭合时跳过这段重复根分离，继续整数搜索。

受限列 MISOCP 在已生成路径上联合优化选址、柜档、分配、航线及运营流量，经过矩阵与物理重建验证后才改善上界。局部搜索限制设施、柜档和分配距 incumbent 的二元 Hamming 距离，默认半径 8，航线仍可自由调整。两者的求解器下界均不作为完整路径问题的全局下界，邻域约束也不进入全局主问题。已验证 incumbent 通过目标截断行收紧主问题的改进区域。受限 MIP、局部搜索、初始设计 CG 和旧邻域 CG 共享默认 20% 的启发式预算；受限 MIP 开启时不再调用旧邻域 CG。

Pareto 默认基础间隔为 10 次整数 CG，单次预算上限 30 秒、累计预算比例 5%。实际预算参考普通对偶耗时并在超时后增加；超时不再指数降低尝试频率，其他失败仍可退避。失败时保留普通认证割，不放宽证书容差。根节点、启发式和 Pareto 的预算比例以进入 Benders 时剩余的求解预算为基数，并始终受总体截止时间限制。开启某项功能不表示它在本次运行产生了有效割、改善解或速度收益。

`support_complete=true`、`catalogue_complete=false` 与 `pricing_complete=true` 可以同时成立：公共模型支持已完成，没有物化所有路径，但定价已遍历并认证隐式路径域。结果只保存生成列，较早的 incumbent 在列池扩张后为新列补零。该变化不保证所有实例提速。本版支持 `K<=2` 和直达飞行，各种多跳或更长任务仍显式拒绝。

独立 `continuous-probe`、`continuous-matrix-probe` 和 `bdd-probe` 构建目标用于连续 CG、同矩阵 CPLEX/Gurobi 复核及 BDD 割质量试验。BDD 探针显式使用完整路径内问题，分别比较连续、设施/柜档整数、全部战略整数三种域，报告截距变化及同一精简根主问题的下界变化；未接入默认生产算法。其条件与用法见 [BDD](../docs/bdd.md)。

`bdd-multiplier-probe` 验证任意乘子的完整域下界和整数可行上平面；`bdd-search-probe` 用盒内 Kelley LP 搜索乘子并反复重解精简根主问题。`fixed` 仅是给定乘子池对照，`search` 不是原文近端 QP 的完整复现。两者均为独立诊断，输出根下界、整数上界、全部割/乘子、阶段耗时及增量日志。实验调度见 `python -m src.experiments.bdd --help`，理论和计费边界见[乘子搜索研究说明](../docs/bdd_multiplier_research_20261022.md)。

固定同一个设计，可以直接比较 CG 与完整 SOCP：

```sh
cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 2 \
  --fixed-design outputs/cpp_cplex_20261016/full_gurobi/K2/solution.json \
  --method benders_cg --cg-max-columns 512 --seconds 300 \
  --out outputs/NEW_CG_K2
```

改为 `--method full_path` 并使用另一个输出目录，即为同设计全列对照。固定设计的 `bound` 只认证该设计下的最优运营成本加固定费用，不能作为选址 MIP 的全局下界。

加速功能的状态以每次运行的 `result.json.accelerations` 为准。开关表示该功能是否启用；实际调用、生成割或改善解的次数需结合 `acceleration_statistics`、`acceleration_records` 和 `iterations` 判断。

| 字段 | 含义 |
|---|---|
| `column_generation` | 在固定设计子问题中增量加入负约化成本列；`benders_full` 为 false。 |
| `reuse_columns` | 将先前子问题发现的服务/空飞列用于后续设计。 |
| `reuse_dual_lp` | 在同次 CG 内增量同步射线 LP 并复用 simplex basis；每轮仍重新认证原锥对偶及完整定价。 |
| `pareto_cuts` | 尝试在普通对偶最优面上改善核心点处的割；辅助求解、完整定价和数值证书通过后才记为接受。 |
| `auxiliary_valid_cuts` | 辅助对偶未通过严格 Pareto 条件、但通过全局有效性及已知原始上界检查且核心点有收益时，单独保留普通有效割。 |
| `courier_savings_bound` | 将直接骑手基准与可实现节约的界加入主问题。 |
| `branch_covering_seeds` | 在少量贪心列之外补齐必要分支，记录缺失与预算中断，不改变公共模型。 |
| `resource_bounds` | 加入静态解析柜容量节省割；依赖 `courier_savings_bound`。 |
| `capacity_cuts` | 在外部最大节省网络中分离最小割，加入仅含设计变量与 θ 的容量成本割；独立于 `courier_savings_bound`。 |
| `resource_lp_cuts` | 外部加权资源 LP 的认证对偶成本割，保留每类服务的收益与容量占用，不增加主问题变量。 |
| `resource_airline_bounds` | 在外部资源 LP 加入配送柜到达、基地出发/返回航线门控；依赖 `resource_lp_cuts`。 |
| `adaptive_master` | 分开内层主问题与外层 gap，限时产生候选，停滞后收紧精度；LB 始终来自 master best bound。 |
| `neighborhood_cuts` | 验证受限模型解池的多样战略，经过完整 CG 认证后生成额外全局割；依赖 `restricted_mip`。 |
| `root_cuts` | 在连续主问题的分数设计处运行认证 CG 并生成普通最优性割。 |
| `root_projection` | 即使外部连续总成本界已闭合，仍检查当前分数候选的运营费用低估。 |
| `root_inout` | 先在合法内点混合设计处分离，必要且剩余预算允许时回到原候选；依赖 `root_projection`。 |
| `restricted_mip` | 在已生成列上联合优化设计和流量，只用验证后的可行目标改善上界。 |
| `local_branching` | 在受限列模型中进行设计邻域搜索；依赖 `restricted_mip`。 |
| `incumbent_cutoff` | 用已验证 incumbent 的总目标收紧主问题目标上限。 |
| `initial_design` | 正式外层迭代前评估一个初始设计。 |
| `warm_start` | 使用已知可行设计启动后续主问题。 |
| `incumbent_cleanup` | 对当前可行解进行模型保持的设计清理；以运行实际记录为准。 |
| `primal_search` | 原有邻域设计 CG 搜索；仅在 `restricted_mip` 关闭时调用，以实际记录为准。 |

`pareto_attempts` 与 `pareto_cuts` 分别是尝试次数和接受割数，不能混用。Benders 的 `pricing_complete` 表示最后处理的子问题已完整定价并满足证书条件；`pricing_complete_scope`、`last_subproblem_stage` 与 `last_subproblem_termination_reason` 明确其范围。它不表示整个 MIP 已达到目标间隙，也不表示尚未处理的新主问题设计已完成定价。

新增投影分离的数学条件、数值恢复、开关与文献依据见 [投影加强](../docs/projection_strengthening.md)。`--reuse-dual-lp true|false` 显式控制 LP 状态复用；`--root-inout-weight 0.9` 控制当前候选在混合点中的权重。`auxiliary_valid_cuts` 不计入 `pareto_cuts`。整数候选统计另存不同设计数、重复设计数及全骑手候选数；warm start 只在当前最好且通过验证的可行方案实际变化时重新提交。

容量割与 CG/Benders 割按同一候选联合生成，再重解主问题；启用 `root_cuts` 时，容量割不跳过该轮分数设计 CG。根轮次仍受时间预算、最优性终止和认证条件约束。中断后可以续算的整数 CG 保持原候选，不重复分离容量割。Pareto、受限 MIP 和局部分支频率按实际整数 CG 调用计数。`cut_schedule` 保存该规则，`candidate_id` 关联割和候选处理记录；每次处理的 `capacity_cuts_added`、`benders_cuts_added` 与统计中的 `root_master_solves`、`root_cg_calls`、`integer_master_solves`、`integer_cg_calls` 可用于核对调度，`master_starts` 仅表示 MIP start 次数。

默认启用上述适用的加速功能。`--plain` 同时关闭可选加强，仍保留规范 CG、必要的物理检查及对偶/定价认证；或用 `--disable NAME` 分别关闭上表除 `column_generation` 外的功能。关闭 `courier_savings_bound` 同时关闭两条静态解析割，动态 `capacity_cuts` 可独立开关，关闭 `restricted_mip` 同时关闭其局部搜索。

预算和频率可通过 `--capacity-cut-limit 100 --root-cut-rounds 20 --root-budget-fraction 0.1 --heuristic-budget-fraction 0.2 --local-branching-radius 8 --pareto-frequency 10 --pareto-budget-fraction 0.05 --pareto-call-seconds 30` 设置。新参数为 `--master-initial-gap 0.05 --master-slice-seconds 30 --resource-lp-seconds 5 --resource-lp-cut-limit 100 --heuristic-pool-size 4 --neighborhood-cut-seconds 15 --neighborhood-cut-budget-fraction 0.05`；各加强可独立关闭，`--plain` 也关闭新增加强。`strengthening_parameters` 保存实参，`master_structure` 保存主问题规模。`master_solves`、`root_diagnostics`、`resource_lp_attempts`、`resource_lp_cuts` 和 Pareto 诊断保存阶段证据，详见 [下界方法及适用性](../docs/lower_bound_research.md) 与 [实现说明](../docs/master_strengthening.md)。`root_cuts` 和 `neighborhood_cuts` 都是 `ordinary_cuts` 的子集，不能重复相加。

## 输出文件与指标

```text
result.json           求解概览、上下界、状态、阶段耗时和算法记录
solution.json         全部决策数组，按实例索引顺序排列
path_flows.json       已登记路径、资源系数及其流量，包括零流量；CG 为生成子集
input_reference.json  输入目录、manifest、model 和本次 K
metrics.json          成本、服务量、任务量和无人机资源汇总
costs.csv             分项小时成本
sites.csv             每处基地/柜点的配置、流量和资源
od_flows.csv          各 OD 需求、服务、第一/第二单流量和收餐时长
links.csv             开通或实际使用的航线及流量
path_flows.csv        所有非零路径流量及每任务时长、成本
OUTPUT_README.txt     输出口径摘要
```

所有需求、订单流量和任务流量均按小时计；费用率为元/h。路径 `unit_cost_cny` 是元/任务，单任务可能服务一单或两单。输入时间系数和 JSON 路径时间为小时；CSV 中名称以 `_minutes` 结尾的字段已转换为分钟。源坐标为 `local_km`，不能将其标注为已确认的经纬度坐标系。

| 指标 | 定义 |
|---|---|
| `objective` | 当前可行解总成本，含基地、柜子、航线、无人机和配送运营成本。 |
| `bound` | 当前用于报告的总目标下界；与 `objective` 一起界定尚未闭合的最优值区间。 |
| `gap` | `(objective-bound)/max(1,abs(objective))`；查看时同时检查求解状态及验证结果。 |
| `served` / `served_orders_per_hour` | 无人机系统服务的订单率；两单任务按两单计入。 |
| `single_tasks_per_hour` / `double_tasks_per_hour` | 一单/两单服务任务的发车率；两者不能直接相加作为订单率。 |
| `courier` / `courier_orders_per_hour` | 直接地面骑手完成的剩余需求。 |
| 基地 `rho` | 基地无人机可用概率决策。柜点没有该决策，相应单元格留空。 |
| `idle_drones` | 平均空闲无人机数量。 |
| `service_busy_drones` / `empty_busy_drones` | 服务/调位任务率乘以任务时长，即每小时忙碌无人机小时数，数值上为平均忙碌架数。 |
| `flight_busy_drones` / `waiting_busy_drones` | 忙碌架数的飞行与任务内等待分解，不能再与服务/空飞忙碌数相加。 |
| 基地 `busy_drones_by_departure` | 按任务出发基地归属的忙碌架数，用于核算；不是基地独立拥有的整数机队。 |
| `accounted_drone_equivalents` | 该基地空闲架数加按出发地归属的忙碌架数。 |
| `fleet` | 全系统连续等效无人机数量；当前模型没有将其改为整数，也没有在成本计算时向上取整。 |
| `fleet_slack` | 全局 `fleet` 减去总空闲、服务忙碌和空飞忙碌架数。 |
| 柜点 `compartments` | 选择的容量档对应的总格口数量。 |
| `mean_occupancy` | 预留与占用格口的平均合计件数；不是每小时吞吐量。 |
| `occupancy_limit` | 选中容量档及溢出概率要求对应的泊松平均占用上限。 |
| `order_pickups_per_hour` / `order_deliveries_per_hour` | 柜点承担的取餐/送餐订单率。 |
| `task_departures_per_hour` / `task_returns_per_hour` | 基地完整任务的出发/返回率，包含服务及调位。 |
| `arc_departures_per_hour` / `arc_arrivals_per_hour` | 经过该点的飞行弧遍历率，与完整任务率不同。 |
| OD `first_order_per_hour` / `second_order_per_hour` | 该 OD 在任务中作为第一单/第二单被服务的订单率。 |
| `mean_receipt_minutes` | 已服务订单从自身请求到最终收餐的流量加权平均时长；没有服务量时留空。 |

`path_flows.csv` 保留所有非零值，包括极小数值流量，不用静默截断来改变解。JSON 路径记录用于重建资源及独立校验，不要求保存未生成路径。`incumbent_kind="all_courier_baseline"` 表示输出的是已知全地面骑手基准；它不能被描述为求解器找到了改进设计。

完整 MIP 入口在求解器候选解与零设施全骑手基准之间保留较低成本者，并另外记录 `solver_candidate_objective`；`has_solver_solution=true` 仅说明求解器返回过可行点。固定设计入口必须保持指定设计，不使用零设施基准替代，没有可行点时会报错。

## 耗时口径

`--seconds` 覆盖输入读取、支持预处理或路径枚举、贪心初始化、模型构造与求解；最终结果整理和文件写出可能使 `wall` 超过给定预算。时间检查在算法阶段边界执行，不保证操作系统层面的硬截止。Python 独立复核及画图属于另行执行的后处理，不计入原生求解时间。

| 字段 | 范围 |
|---|---|
| `input_read_seconds` | CSV/JSON 解析、SHA 校验和输入合法性检查。 |
| `enumeration_seconds` | 全路径基准的目录构造；隐式 CG 为 0。 |
| `support_preprocessing_seconds` | 隐式 CG 首段/分支支持构造。 |
| `timings.greedy_initial_columns` | 贪心种子与分支覆盖补充路径构造。 |
| 完整模型 `timings.build` | 数学模型记录构造、Concert 对象构造与 CPLEX 提取。 |
| 完整模型 `timings.optimize` | CPLEX `solve()` 调用耗时。 |
| 完整模型 `timings.total` | 完整模型函数全过程，还包括解读取、残差检查和内存结果整理；不含磁盘写出。 |
| Benders `timings.master_build` | 主问题记录、加强约束及求解器对象构造。 |
| Benders `timings.master_optimize` | 外层主问题求解累计时间。 |
| Benders `timings.master_update` | 主问题启动值与割更新累计时间。 |
| Benders `timings.root_master` | 根节点连续主问题构造与求解；根节点运营 CG 计入子问题阶段。 |
| Benders `timings.capacity_cut_build` | 从完整输入构造货币最大流网络。 |
| Benders `timings.capacity_cut_separation` | 最大流、原图割容量累计、去重和新增容量行的累计时间。 |
| Benders `timings.restricted_mip` | 受限列 MISOCP 与局部搜索的构造、求解及验证累计时间。 |
| Benders `timings.subproblem_wall` | 各次固定设计子问题完整过程累计时间。 |
| Benders `timings.subproblem` | 子问题阶段累计：`build`、`primal`、`dual`、`pricing`、`add_columns`、`pareto`、`read_solution`、`validation`、`certificate`。 |
| Benders `algorithm_wall` | Benders 函数整体墙钟时间，包含算法控制与结果整理。 |
| `output_seconds` | 原生 JSON/CSV 结果写出时间。 |
| `wall_before_output` / `wall` | 从开始读取输入到写出前/写出后的总体墙钟时间。 |

`subproblem_wall` 包含子阶段计时，`timings.total`、`algorithm_wall` 和 `wall` 同样包含各自子阶段，不能把包含关系中的项目重复求和。最终 `result.json` 补写耗时字段的极小开销不在 `wall` 的采样终点内。

子问题及 Benders 函数内部的计时采样发生在局部求解器对象析构之前；对象释放和未单独计量的控制开销归入主入口总时间的余项。端到端比较使用 `wall`，不能把子阶段之和当作总时间。

## 与 Python / Gurobi 对照及独立复核

固定 TSL 实例已完成三个层次的对齐检查：

1. 原生与 Python 标准读取器的 25 个输入数组逐值一致，K=1/K=2 都通过。
2. 每条路径的身份、顺序、整数属性和资源系数一致。
3. K=1/K=2 分别测试完整模型、全零固定设计和另一组二进制固定设计，共 6 种模型。2026-10-18 的隐式版本将航段激活界改为输入结构界；测试独立验证新旧界及其差异，再逐项比较其余变量、目标、线性系数/RHS/重复次数和锥。新旧分数松弛不保证相同，固定二元设计的运营可行域保持一致。

这些检查核对相同物理模型，并明确标记用于分数松弛的航段界差异。**同一时间预算下，CPLEX 和 Gurobi 的搜索路径、当前可行解、下界和间隙不要求相同。** 对比应同时报告可行成本、认证下界、间隙、终止原因、服务量和验证状态；若尚未达到相同精度，不能只凭一个成本值判定实现不一致或算法更优。还可把同一个二进制设计交给双方求固定设计 SOCP，排除离散搜索路径的影响。即使最优目标相同，多重最优解也可能导致点位或路径流量不同。

原生结果的 Python 独立验证：

```sh
python -m src.experiments.native validate \
  outputs/cpp_tsl20h_full_K2 --instance examples/tsl_2l20h50od --k 2
```

该步骤重新检查路径物理资源和运营约束，在运行目录下写出 `native_validation.json` 和 `operations/`。它属于独立验收，不是原生求解器的运行依赖。

Gurobi 完整模型参考：

```sh
python -m src.experiments.native gurobi \
  --instance examples/tsl_2l20h50od --k 2 \
  --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/gurobi_tsl20h_full_K2
```

在参考命令中增加 `--fixed-design outputs/cpp_tsl20h_full_K2/solution.json`，即可固定同一设计进行互验。双方参考运行的预算范围均覆盖读取、枚举、构造和求解；验证及写出耗时分别报告。

原生读取和路径目录回归：

```sh
make -C cpp input-probe path-catalogue -j2
python cpp/tests/input_reference.py --binary cpp/build/mlnd_input_probe \
  --output outputs/cpp_checks/input_verification.json
python cpp/tests/path_reference.py --binary cpp/build/mlnd_path_catalogue \
  --output outputs/cpp_checks/path_verification.json
```

数学模型回归使用 `cpp/tests/formulation_probe.cpp` 和 `cpp/tests/formulation_reference.py`；它只构造并比较矩阵，不运行优化。各项比较的 JSON 证据与实际求解结果分开保存，避免把矩阵等价检查写成最优性证明。

完整模型矩阵、Benders 和固定设计边界回归：

```sh
make -C cpp formulation-probe benders-probe fixed-design-probe -j2
python cpp/tests/formulation_reference.py --binary cpp/build/mlnd_formulation_probe \
  --output outputs/cpp_checks/formulation_verification.json
python cpp/tests/benders_reference.py --binary cpp/build/mlnd_benders_probe \
  --output outputs/cpp_checks/benders_verification.json
python cpp/tests/fixed_design_reference.py --binary cpp/build/mlnd_fixed_design_probe \
  --output outputs/cpp_checks/fixed_design_verification.json
```

标号和 CG 的专项验收：

```sh
make -C cpp pricing-probe benders-probe -j2
cpp/build/mlnd_pricing_probe examples/tsl_2l20h50od outputs/cpp_checks/pricing.json
python cpp/tests/benders_reference.py --output outputs/cpp_checks/benders.json
# 已保存 fixed_cg/K1,K2 与 full_cplex/K1,K2 的同设计实验后：
python cpp/tests/cg_reference.py --runs outputs/implicit_cg_20261018/verified
```

`pricing_probe` 比较任意对偶、已选列和上界价格下的最小约化成本，涵盖非对称航线、AA、重访及资源边界；`benders_reference` 还验证每次仅加一列的收敛、零预算无割、微型实例连续根界等价性和零目标 gap 下的数值停止行为。`cg_reference` 读取保存结果并独立重建物理资源，不重新求解。

主问题加强、分支覆盖与联合启发式的专项验收：

```sh
make -C cpp strengthening-probe branch-seed-probe heuristic-probe -j2
cpp/build/mlnd_strengthening_probe examples/tsl_2l20h50od 1 outputs/cpp_checks/strengthening_K1.json
cpp/build/mlnd_strengthening_probe examples/tsl_2l20h50od 2 outputs/cpp_checks/strengthening_K2.json
cpp/build/mlnd_branch_seed_probe examples/tsl_2l20h50od > outputs/cpp_checks/branch_seed.json
cpp/build/mlnd_heuristics_probe > outputs/cpp_checks/heuristics.json
```

`strengthening_probe` 以完整参考路径逐条验证成本/资源分摊、逐订单交付柜系数和主问题无运营辅助变量；还检查局部容量瓶颈、6561 次分数设计跨割投影、不同列池的加强矩阵相同、全骑手方案不被排除、断连与零占用边界；可追加同实例的参考 `result.json`，验证真实可行解对加强主问题的投影。`branch_seed_probe` 检查 aggregate 分支覆盖、prefix 共用首段、分配掩码、物理缺失、去重及预算中断。`heuristics_probe` 检查可行上界、半径零邻域、无效 incumbent/路径系数拒绝、空列池与内层下界不泄漏。参考枚举仅用于测试；这些检查不能代替同预算的算法性能实验。

下界诊断与邻域候选池检查：

```sh
make -C cpp cut-pool-probe root-diagnostic-probe -j2
cpp/build/mlnd_cut_pool_probe > outputs/cpp_checks/cut_pool.json
cpp/build/mlnd_root_diagnostic_probe examples/tsl_2l20h50od 1 180 \
  outputs/cpp_checks/root_K1.json --root-rounds 100
cpp/build/mlnd_root_diagnostic_probe examples/tsl_2l20h50od 2 180 \
  outputs/cpp_checks/root_K2.json --root-rounds 100
```

根诊断为全路径连续模型和普通 Benders 根阶段分别分配上述预算，不能把命令中的 180 秒误读为两者合计。它关闭其他主问题加强以单独检查普通投影，记录原矩阵残差、连续上下界、停止原因及同模型数值重求。参考无效或根阶段未收敛时不声明等价性。输出路径必须未使用。`cut_pool_probe` 检查不改善上界的有效候选仍可入池、去重与池上限、半径零、关闭开关及局部下界不泄漏；外部资源 LP 的全域有效性、零占用与预算回退由 `strengthening_probe` 检查。

固定设计边界测试包括：无设施零服务；分配订单但不开航线且零服务；违反基地分配约束；求解预算耗尽且没有可行点。后两种不会返回一个与指定设计不同的全骑手解。Benders 回归还把生成的割放到其他设计下逐项检查，覆盖有序双单、两种任务流策略、容量变化和关闭航线。

本轮 TSL 实现验收保存在 `outputs/cpp_cplex_20261016/`，仅用已有结果重新生成对照报告：

```sh
python -m src.reporting.native outputs/cpp_cplex_20261016
```

报告列出原始冷启动结果、固定设计互验、独立热启动补充，以及每基地、柜点和 OD 的指标。历史运行的二进制与包装器差异保存在 `experiment_protocol.json`。正式完整模型预算为 600 秒，Benders 预算为 120 秒；这是迁移验收，不能把它们直接排为同预算性能优劣。TSL 上 Pareto 的实际尝试、接受和拒绝原因需查看记录；小算例上的有效割验收不保证大算例强化求解稳定或更快。

## 源码职责

```text
include/mlnd/      输入、路径、模型、求解与输出接口
src/input.cpp     标准实例读取与校验
src/paths.cpp     物理可行路径和资源系数
src/formulation.cpp  完整模型与固定设计模型的共同矩阵
src/linear_model.cpp CPLEX Concert 适配、求解和残差检查
src/benders.cpp   主问题、规范 SOCP 列生成、对偶证书及割
src/strengthening.cpp  输入解析成本/资源割与外部最大流容量成本割
src/heuristics.cpp 已生成列上的联合 MISOCP、局部搜索及独立可行性检查
src/pricing.cpp   精确多基地服务阶段标号定价及后续扩展支配
src/output.cpp    原生结果打包与指标输出
src/main.cpp      命令行和端到端执行
tests/            输入、路径、模型等价回归
```

构造路径和模型时不重新拟合需求、不更换 κ，也不重新解释固定资产折旧。原始数据、论文和冻结基准不属于此实现的写入对象。

两种需求实现均支持 `--structural-strengthening`，开启分配关联聚合、共同 ρ 联合线性化和排队 perspective 三项加强；默认关闭以保留旧版对照。它们保持整数模型不变，强化两种方法共享的连续根松弛，主问题仍不增加运营辅助变量。独立开关、数学依据和验证范围见 [结构加强说明](../docs/structural_strengthening.md)。

四组算法的新旧比较使用 `python -m src.experiments.structural_benchmark`，冻结修改前后的二进制与同一 TSL 输入。每个 K 分别比较旧/新 lifted 与旧/新 conditional；连续根消融独立运行，不作为免费初始化传给算法。回归入口为 `make structural-strengthening-probe structural-cg-cut-probe`。
