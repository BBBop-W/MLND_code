# MLND consecutive delivery network

当前代码采用**任务级独立指派**：每个单订单或有序双订单任务独立选择出发基地，返回基地由路径选择。时限按骑手送到家时间判断，需求重构规则保持原样。完整 CPLEX、Benders、定价认证和上界搜索共享这一模型；没有旧首单捆绑模型的并行开关。

L2 K2 的整合入口为 `--preset performance-20261105`，包含现有加速改动。它们已完成小算例正确性验证，大规模性能结论仍以之后同模型实验为准。K1 或其他基地数需要关闭当前限定于 L2 K2 的 ALNS 相关开关。用法见[当前算法说明](docs/current_baseline.md)，迁移、证明要点和验证见[任务指派迁移说明](docs/task_assignment_migration_20261105.md)。

历史日期 preset 保留其加速参数组合，但在当前源码中统一使用新任务模型；复现旧实验必须使用该批冻结源码和二进制。已有实验输入和冻结程序不被本次修改替换。

## 构建与运行

Windows 服务器从[安装、构建与验收说明](docs/server_runbook_20261112.md)开始；当前项目评估、修复和数值实验顺序见[2026-11-12 审查](docs/project_assessment_20261112.md)。服务器只需要 CPLEX；独立 Python 校验不要求 Gurobi，历史 Python 优化器按需安装 `.[gurobi]`。

需要 C++17、CPLEX Concert 和 `nlohmann/json.hpp`；原生读取、求解、输出不依赖 Python。

```sh
make -C cpp -j1 BUILD_DIR=build-current

# 当前 L2 K2 算法；K1 的适用配置见 docs/current_baseline.md。
cpp/build-current/mlnd_cpp --preset performance-20261105 \
  --input examples/tsl_2l20h50od --k 2 \
  --seconds 600 --gap .01 --threads 1 --seed 43 \
  --out outputs/NEW_BASELINE_RUN

# 同物理模型的完整路径 CPLEX 对照。
cpp/build-current/mlnd_cpp --preset full-cplex-20261029 \
  --input examples/tsl_2l20h50od --k 2 \
  --seconds 600 --gap .01 --threads 1 --seed 43 \
  --export-catalogue --out outputs/NEW_FULL_CPLEX_RUN

# 独立重算可行性，不调用优化器。
python -m src.experiments.native validate outputs/NEW_BASELINE_RUN \
  --instance examples/tsl_2l20h50od
```

使用新的输出目录。preset 放在参数前面，后续参数可覆盖其设置；精确生效配置保存在结果中。完整 CPLEX 与算法共享固定需求、联合可用率加强、队列 perspective 和航线流量界；算法的附加割与搜索属于算法本身。

两种方法当前都预处理完整物理路径目录。Benders 在 SOCP 中只加入所需列，并通过完整域定价认证；这不等于无需保存完整目录。`--export-catalogue` 在输出收尾阶段额外流式保存完整目录，供割证书独立重放；通常的可行解输出仍保持紧凑。

正常构建只生成生产求解器。诊断按需构建，例如 `make -C cpp queue-fractional-probe`；BDD 研究代码只链接到显式诊断目标，不进入生产求解器。构建与参数细节见 [cpp/README.md](cpp/README.md)。

## 数值实验与历史结果

当前服务器入口为 `scripts/run_server_experiment.py`：一次运行一个实例，固定二进制、输入和源码指纹，分别保存求解与独立物理校验状态。`scripts/check_server.py` 先执行小型正确性验收。实验使用新的输出目录，按相同实例、seed、线程、时间预算比较当前任务模型的完整 CPLEX 与 Benders；达到目标 gap 仍须保留实际终止原因。

[2026-10-29 比较](docs/baseline_cplex_results_20261029.md)及[结果索引](docs/algorithm_results.md)属于对应历史模型和冻结批次，不作为当前独立任务模型的性能结论。历史日期实验脚本保留用于复现，不作为 Windows 新批次的启动入口。

## 模型与输入

需求在网络设计之前固定重构，保留剩余单单并逐 OD 守恒；每个任务独立选择出发基地和首段。K=1 配对流量为零。历史 `--first-leg-consistency` 会重新耦合任务，当前入口明确拒绝。模型假设、有效性范围和源码索引见：

- [当前任务模型与配置](docs/current_baseline.md)及[迁移推导](docs/task_assignment_migration_20261105.md)
- [代码结构与职责](docs/architecture.md)；[2026-11-05 算法说明](docs/algorithm_review_20261105.md)是迁移前快照
- [2026-11-05 问题修复、验证与后续加速方案](docs/fixes_and_acceleration_plan_20261105.md)
- [历史 lifted 模型说明](docs/model.md)（主要为早期 K2 基地对指派与指派相关需求重构，供对应版本复现）
- [路径支持割](docs/path_support_validity_20261027.md)、[任务成本加强](docs/task_cost_strengthening_validity_20261027.md)
- [队列割设计](docs/l2_queue_polymatroid_cut_design_20261028.md)与[分数点分离](docs/l2_fractional_queue_implementation_20261029.md)

原生入口读取 `model.json`、CSV 与 manifest，例如 `examples/tsl_2l20h50od/`；Python 历史优化入口读取 `input_metadata.json` / `input_arrays.npz`，两种目录不能互换。Python Gurobi 优化入口尚未实现当前固定广义任务模型，不能把它作为本轮同模型求解器对照。新实例构造见[规模阶梯](docs/scale_ladder.md)和[真实地图协议](docs/unified_benchmark_20261027.md)。

## 修改和复核入口

| 工作 | 共享入口 |
|---|---|
| 当前版本配置、CLI | `cpp/src/solver_presets.cpp`、`cpp/src/main.cpp` |
| 输入与固定任务预处理 | `cpp/src/input.cpp`、`cpp/src/preprocessed_demand.cpp` |
| 物理路径与系数 | `cpp/src/paths.cpp`；枚举和定价共享资源构造 |
| 完整模型与原生求解接口 | `cpp/src/formulation.cpp`、`cpp/src/linear_model.cpp` |
| Benders、定价、上界搜索 | `cpp/src/benders.cpp`、`pricing.cpp`、`heuristics.cpp` |
| 队列割及分数分离 | `cpp/src/queue_polymatroid.cpp`、`queue_fractional.cpp` |
| 保存与运行进度 | `cpp/src/output.cpp`、`progress.cpp` |
| 当前服务器运行、验收 | `scripts/run_server_experiment.py`、`scripts/check_server.py` |
| 历史冻结运行、割审计 | `src/experiments/baseline_benchmark_20261029.py`、`baseline_cut_audit_20261029.py` |
| 独立运营重算 | `src/experiments/native.py`、`src/operations/validation.py` |

修改数学约束时复用公共模型与路径系数，保持完整模型、子问题、定价和输出一致。受限列目标、局部 MIP 界和候选解值不能替代原问题全局 LB。

## 历史入口与归档

`lifted` / `conditional`、旧基地对分配、Python 优化器及 BDD 诊断保留用于对应版本复现。日期文档和冻结批次描述当时的模型与配置，不代表当前默认。历史资料见[算法历史](docs/k1_algorithm_history_20261026.md)、[原生使用说明](cpp/README.md)和[结果索引](docs/algorithm_results.md)。原始订单、论文、旧实验和旧源码快照不参与本轮覆盖或删除。
