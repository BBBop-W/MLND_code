# K=1 算法工作入口与历史试验索引

整理日期：2026-10-26。
本轮仅保存分支、整理现有结果并检查工作入口，没有启动新的性能对照试验。

## 分支与模型范围

| 分支 | 用途 | 基线/保存提交 |
|---|---|---|
| `codex/generalized-demand-20261026` | 保存需求预处理 K=2、共同出发基地、首段一致性、验证和实验代码 | `cfa78413be1c9c9d28dd8ea200330094cf87ccd1` |
| `codex/k1-origin-assignment` | 原三下标 K=1 基线，保持原指针 | `da5591f52965e06decb470c7b627311e91e123f9` |
| `codex/k1-algorithm` | 后续 K=1 算法工作，当前分支 | 从 `da5591f` 建立 |

当前 K=1 使用 `alpha[r,l]`，只指定 OD 的出发基地；服务路径可以返回不同基地，保留基地流量平衡及空载调拨。
它与历史 `alpha[r,l,m]` 基地对模型并非等价，历史界和设计文件不能直接混用。
数学口径及已适配的算法见 [起点分配说明](k1_origin_assignment.md)。

新分支不包含保存分支中新增加的 `preprocessed` 原型，但不破坏性删除基线中的 K=2 公共代码。
所有新算法实验显式指定 K=1；标准实例本身可能默认 K=2。
`outputs/`、`cpp/build*/` 均被 Git 忽略，切分支不会删除现有实验及构建结果。
保存分支的实验仍位于 `outputs/generalized_demand_L01_20261026/` 和 `outputs/generalized_first_leg_L01_20261026/`。
以下报告链接指向本地保留的输出，不随 Git 提交复制；本索引中的关键数值随代码保存。

## 首选基准：当前三下标 K=1

来源：[完整报告](../outputs/k1_origin_comparison_20261024/REPORT.md)、[CSV](../outputs/k1_origin_comparison_20261024/comparison.csv)、[来源与命令](../outputs/k1_origin_comparison_20261024/manifest.json)。
实例为 TSL 2 基地 / 20 柜候选点 / 50 OD，完整路径包括 1101 条服务路径和 2 条空载路径。
各组预算 600 秒，单线程，seed=43，目标 gap=0，aggregate/direct，不使用历史热启动和历史下界。
本次整理核对了该 CSV 引用的七份结果文件哈希，与保存记录一致；原报告保存了独立运营核验结果。

| 模型/方法 | UB（元/小时） | LB（元/小时） | gap | 实际求解秒数 | 终止 |
|---|---:|---:|---:|---:|---|
| 当前三下标 Benders-CG | 9969.265139 | 9752.394647 | 2.1754% | 600.212 | time_limit |
| 当前三下标 CPLEX 全路径 | 9969.265139 | 9964.453599 | 0.0483% | 600.102 | AbortTimeLim |
| 当前三下标 Gurobi 全路径 | 9969.265140 | 9969.265140 | 0% | 18.688 | optimal |
| 相邻四下标 Benders-CG | 9972.608755 | 9731.304855 | 2.4197% | 600.491 | time_limit |

当前 Benders 找到了与独立 Gurobi 最优值在数值容差内一致的可行解，但自身认证下界没有超过连续根界，未证明最优。
当前连续根界为 9752.394745，数值保守调整后采用 9752.394647；最终 LB 与之相同。
主问题耗时约 349.919 秒，777 条外层记录，804 条普通割、20 条根割、24 条有效辅助割、100 条资源 LP 割，记录的最大主问题分支节点数为 0。

模型规模：

- 分配二元由 200 减为 100。
- 当前完整模型 3864 个变量、4189 条线性约束、2 个锥；相邻旧模型为 4064 个变量、4889 条线性约束、2 个锥。
- 当前 Benders 主问题有 706 个设计二元及一个运营成本 epigraph，共 707 个变量。
- 生成列数是算法实际使用的子集，不能当作完整路径数量。

## 先前 K=1 算法强化：均为旧基地对分配

这些实验用于回顾思路，不与当前三下标结果混成同模型的速度对照。

| 阶段 | 预算/范围 | 主要结果 | 原报告 |
|---|---|---|---|
| 连续根 CG | 2L/20H/50OD，180 秒 | LB 7405.6868 → 9731.3051；gap 25.7154% → 2.4263%，主要改善来自 LB | [root CG](../outputs/root_cg_bdd_20261021/REPORT.md) |
| 资源投影割、对偶 LP 复用 | 同实例，600 秒 | K1 LB 仍为 9731.305；旧新 gap 2.418% / 2.427%，该轮未改善 K1 最终结果 | [projection](../outputs/projection_cuts_20261021/REPORT.md) |
| OD-airline 单项开关 | 同实例，600 秒 | K1 lifted 开/关 LB 均为 9731.305，gap 均约 2.435% | [OD-airline](../outputs/od_airline_benchmark_20261022/analysis/REPORT.md) |
| 三项结构加强 | 同实例，600 秒 | assignment incident clique、shared rho joint、queue perspective；lifted LB 9731.305 → 9756.729，gap 2.410% → 2.176% | [structural](../outputs/structural_strengthening_20261022/comparison/analysis/REPORT.md) |
| BDD 乘子搜索诊断 | 同实例，预算 600 秒 | 2000 次内问题调用、954 条割；重求根主问题后 LB 仍为 9731.30486，提升为 0 | [BDD](../outputs/bdd_multiplier_20261022/comparison/analysis/REPORT.md) |

三项结构加强的 +25.424 LB 改善不属于 OD-airline 单独开关。
这些 LB 属于旧基地对模型，不能作为当前三下标模型的有效界直接加载。
BDD 实验是有限乘子盒上的 Kelley 变体，不是完整 BDD3 复现；它没有接入生产可行解启发式，其全骑手 UB 不用于评价完整算法质量。
BDD 达到调用上限提前停止，准备加进程墙钟约 387.54 秒，内部计时约 385.74 秒；它不是足额运行 600 秒的生产算法比较。

## 之前讨论的两个小算例与分解诊断：K=2 历史记录

L01 为 1L/10H/25OD，L02 为 2L/20H/50OD。
下表对应当时的 K=2 基地对分配，保留为算法研究记录。

| 实验 | 原报告 | 结论 |
|---|---|---|
| 完整 lifted vs 条件子问题，L01–L03 | [最终结果](../outputs/conditional_benchmark_20261022/analysis/RESULTS.md) | 180 秒/组、单线程串行；最终 12/12 已结束且报告记录核验通过；条件子问题缩小，但 L01/L02 最终 LB 未超过连续根界 |
| 拓扑及航线界，L01/L02 | [topology](../outputs/topology_benchmark_20261022/analysis/REPORT.md) | 600 秒；根 LB 分别提高约 0.349 / 0.364，整数阶段未继续提高 |
| 分配与路径支持成本割 | [assignment support](../outputs/assignment_support_benchmark_20261022/analysis/REPORT.md) | 600 秒；减少部分无效 SOCP 评估，LB 没有进一步上升 |
| 双单支持与跨基地联合成本割 | [branch joint](../outputs/branch_joint_benchmark_20261022/analysis/REPORT.md) | 600 秒；实际 SOCP 的全骑手候选降为 0，但 LB 仍未上升 |
| 固定网络、保留分配整数性 | [修正后摘要](../outputs/fixed_network_diagnostic_20261024/analysis/SUMMARY.md) | L02 固定网络后两组整数问题仍在 600 秒结束时有约 2.17% / 2.22% gap；这是固定网络诊断，不是嵌套 Benders 完整算法对比 |

最终双单/联合割一轮：

| 实例/方法 | UB | LB | gap | 状态 |
|---|---:|---:|---:|---|
| L01 lifted | 3822.585 | 3755.230 | 1.762% | 600 秒，time_limit |
| L01 conditional | 3823.318 | 3755.230 | 1.781% | 600 秒，time_limit |
| L02 lifted | 9941.256 | 9293.946 | 6.511% | 600 秒，time_limit |
| L02 conditional | 9940.122 | 9293.946 | 6.501% | 600 秒，time_limit |

固定网络诊断应读 `SUMMARY.md`：它综合了连续分配解的数值复查，原 `REPORT.md` 中部分连续结果未通过检查。
固定网络问题的 LB 只用于该固定网络，不能当作原选址模型的全局 LB。

## 曾运行后叫停的规模实验

[2 小时批次报告](../outputs/conditional_overnight_20261022/analysis/REPORT.md) 对应 K=2、L01–L10、每组 7200 秒、4 并发、各单线程。
L01/L02 有完成记录，L03 以后多为内存终止或用户取消，不是完整的规模梯度证据。

旧五小时阶梯的历史记录位于 [20261021 快照](../outputs/tsl_scale_ladder/analysis_snapshots/20261021_171751/completed_runs.csv)。
当前 `outputs/tsl_scale_ladder/analysis/REPORT.md` 显示 0/60、全部 not_run，与历史快照不是同一验收状态。
快照包含构造超时、失败时保留的可行解、中断检查点及计时异常，后续引用需逐行核对，不把它们整理成完整等预算比较。

## 继续 K=1 工作时复用的入口

构建及只读检查：

```sh
make -C cpp -j2 BUILD_DIR=build-k1
cpp/build-k1/mlnd_cpp --input examples/tsl_2l20h50od --k 1 --inspect
```

后续正式比较可使用以下现有入口；本次整理不执行这些性能试验：

```sh
cpp/build-k1/mlnd_cpp --input examples/tsl_2l20h50od --k 1 \
  --method benders_cg --seconds 600 --gap 0 --threads 1 --seed 43 \
  --out outputs/NEW_K1_BENDERS

cpp/build-k1/mlnd_cpp --input examples/tsl_2l20h50od --k 1 \
  --method full_path --seconds 600 --gap 0 --threads 1 --seed 43 \
  --out outputs/NEW_K1_CPLEX_FULL

/opt/anaconda3/bin/python -m src.experiments.native gurobi \
  --instance examples/tsl_2l20h50od --k 1 \
  --seconds 600 --gap 0 --threads 1 --seed 43 \
  --out outputs/NEW_K1_GUROBI_FULL

/opt/anaconda3/bin/python -m src.experiments.native validate outputs/NEW_K1_BENDERS \
  --instance examples/tsl_2l20h50od
```

新输出目录必须不存在或为空，不能覆盖历史结果。
原生 CSV/JSON 实例交给 C++ 或 `src.experiments.native`；Python `main.py run --input` 使用 NPZ/metadata 快照，两种输入入口不混用。
原生结果复核、比较表、阶段时间及绘图继续复用 `src.experiments.native`、`src.reporting.native` 和 `src.reporting.style`，不另起一套画图实现。
公共模型、定价和割的代码地图见 [根 README](../README.md#从哪里开始读和修改) 及 [C++ README](../cpp/README.md)。
`--assignment-support-cuts`、`--double-support-cuts`、`--joint-base-cost-cuts` 当前证书依赖固定返回基地，K=1 会拒绝这些开关；若以后迁移，需重新证明当前分配域上的有效性。

## 本次分支整理检查

- `cpp/`、`src/`、`tests/`、`configs/` 的跟踪文件与基线 `da5591f` 一致，未改模型或算法实现。
- `make -C cpp -j2 BUILD_DIR=build-k1 all build-k1/mlnd_origin_assignment_probe` 编译成功。
- C++ 起点分配探针对 aggregate/prefix 两种输入均通过，验证同一 OD 可返回两个基地、需求及网络约束残差、旧分配维度和不适用割的拒绝。
- 标准实例 `--k 1 --inspect` 返回 `assignment_scope=origin`、100 个分配变量、2L/20H/50OD。
- `test_origin_assignment.py`、`test_native.py`、`test_operations.py` 共 57 项测试通过。
- 本索引 18 个本地链接存在，`git diff --check` 通过。
- 本次没有重新运行历史性能试验，没有修改 MLND 论文或原始订单数据。
