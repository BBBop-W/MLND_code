# 当前算法与使用范围

当前生产实现直接采用任务级独立指派和骑手送到家时限。需求重构的成本节约排序、剩余需求概率与共同出发点筛选规则保持不变；共同支持仅用于需求生成，不再约束不同任务的执行指派。任务率会在新的完整可行路径域上重新计算。

L2 K2 使用 `--preset performance-20261105`，直接整合现有成本/资源割、次模队列割、定价认证、模型缩减、缓存、稳定化和 ALNS。完整 CPLEX 对照使用 `--preset full-cplex-20261029`。它们共享输入、任务流量等式和物理路径域。

```sh
make -C cpp -j1 BUILD_DIR=build-current
cpp/build-current/mlnd_cpp --preset performance-20261105 \
  --input INPUT_DIRECTORY --k 2 --seconds 600 --gap .01 --threads 1 \
  --out outputs/NEW_RUN
```

K1 或非双基地实例使用相同入口时，加上以下参数，关闭当前仅为 L2 K2 实现的上界模块：

```sh
--disable primal_alns --disable primal_support_completion \
--disable indexed_support_completion --disable cg_primal_feedback
```

输出 `assignment_scope="task"`，任务 ID 先排 R 个单任务，再按行排 R² 个有序双任务（K1 仅 R 个）。`task_table` 保存每个 ID 的订单组成和需求率，`task_assignments.csv` 保存独立指派；旧首单指派解不会被静默解释为当前设计。

输入 `consecutive-csv/2` 显式保存 `home_receipt_mean`。历史 CSV/1 只读兼容，由冻结骑手终端参数恢复该系数；混合收件均值不足以识别骑手送到家时间时明确拒绝。柜内移除均值、成本系数和续单窗口规则不改变。

日期 preset 是参数组合，不能单独识别旧模型。复现旧结果必须使用原批次冻结的源码、二进制和输入；本次未覆盖这些材料。历史结果见[结果索引](algorithm_results.md)。

小算例验证、精确认证边界及源码记录见[迁移说明](task_assignment_migration_20261105.md)。当前没有新模型在 L2/L3 大实例上优于 CPLEX 的结论。
