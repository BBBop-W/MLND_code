# 路径结果打包修复与一小时复测

## 修改范围

之前每次返回整数运营子问题结果时，都会把完整物理目录中的路径（含零流）转换为JSON。50km²、4L/40H/100OD算例有4,474,549条服务路径，导致约100秒级包装/清理空窗，最终检查点约7.71GiB。

现在对 `preprocessed + aggregate + !first_leg_consistency` 使用精简物理记录：

- 保留所有 `flow != 0` 路径，包括极小正流和负值，没有新增过滤阈值。
- 每个 `(首单类型, 次单类型, 出发基地)` 保留最低成本路径，因此同时保持全局最低成本和每个出发基地的支持信息。
- 每条航段保留最大遍历次数见证，使原有航段约束校验使用的统计不变。
- 空载路径仍完整保存；全量流量数组、首段数组及原始路径ID不变。
- 缓存见证索引，路径池增长后重算。定价、路径可行性、需求重构和优化矩阵不使用该缓存。
- 其他运营模式继续使用原密集输出。Python独立运营校验无需修改，仍从物理记录重新计算成本、支持和需求重构。

记录集合是完整已枚举目录的充分统计见证，不应将其数量称为完整路径数量。`catalogue`/`path_storage`继续描述求解域；新增 `path_record_storage`/`path_record_counts`描述输出记录。见证路径本身不单独证明全域最小性；充分统计一致性由完整目录筛选及回归核对。

## 验证

证据目录：`outputs/compact_output_validation_20261027/`。

- 73项Python回归通过，覆盖native读写、需求重构、精简输出、批次配置和内存监护。
- C++小例K1/K2的密集与精简输出成本、支持、最大遍历统计及运营结果一致；正负 `1e-250` 流保留，重复ID、越界、路径篡改、流量不一致及漏非零记录被拒绝。
- 小例固定设计SOCP与列生成成本一致，4组共36次跨设计割检查通过。
- 大例4,474,549条服务路径仅需14,448条零流校验记录（本检查使用纯骑手方案）。打包与记录核对0.813秒，写出0.370秒；包含目录生成和重构总计7.15秒。
- 大例 `path_flows.json` 14.69MiB、`solution.json` 39.25MiB；既有Python独立校验通过，校验与运营导出3.39秒。
- 大例无求解器输出检查峰值RSS约4.92GiB，独立校验约1.12GiB；这不是一小时优化的内存预测。

`unchanged_model_sources.json`记录本轮核心模型、需求重构、定价、Benders和求解器源文件与上一轮10分钟冻结版的逐文件哈希一致性。改动在输出、输出缓存、稀疏记录验证和测试中；没有改变数学模型或运营规则。

## 一小时批次

新目录：`outputs/baseline_tsl_A50_L4_H40_OD100_20261027_1h/`。

复用上一轮完全相同的输入和加强策略。K2统一模型；CPLEX全路径先运行，Benders-CG随后运行。各3600秒、单线程、串行、gap 1%、seed43、16GiB单任务/批次RSS上限、2GiB系统可用内存底线。每项外部3720秒上限包含收尾与独立审计，不额外增加优化预算。

准备命令（拒绝覆盖已存在目录）：

```sh
/opt/anaconda3/bin/python -m src.experiments.baseline_capacity_20261027 prepare --batch outputs/baseline_tsl_A50_L4_H40_OD100_20261027_1h --binary cpp/build-compact-output/mlnd_cpp --seconds 3600 --hard-wall-seconds 3720
```

运行、查看与汇总继续使用原冻结入口：

```sh
/opt/anaconda3/bin/python outputs/baseline_tsl_A50_L4_H40_OD100_20261027_1h/run.py launch
/opt/anaconda3/bin/python outputs/baseline_tsl_A50_L4_H40_OD100_20261027_1h/run.py status
/opt/anaconda3/bin/python outputs/baseline_tsl_A50_L4_H40_OD100_20261027_1h/run.py report
```

后台调度完成两项后自动运行独立校验及报告。任务启动不等于完成；最终比较应读取 `state.json` 的终态和 `analysis/REPORT.md`。本轮同时增加时间预算和修复输出，不能把最终求解质量的全部变化归因于输出修复。
