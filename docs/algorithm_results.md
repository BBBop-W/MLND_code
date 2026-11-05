# 当前算法与历史结果索引

本页把推荐入口与各轮冻结证据集中起来。不同批次的时间、UB/LB 和算法版本不拼接成一次运行；日期报告中的“默认关闭”描述该轮开发状态，当前生产推荐由 [baseline-20261029](current_baseline.md) 定义。

| 轮次 | 比较内容 | 结论/状态 | 结果入口 |
|---|---|---|---|
| 2026-10-28 queue EPI | 原算法 vs 队列有效割，L2 M01/M07/M08 | 已完成；生产 LB 收益与割有效性分别报告 | [构造与结果](l2_queue_polymatroid_results_20261028.md) |
| 2026-10-28 price + target | queue 基线 vs 原生价格搜索与目标精化 | 六作业完成，最终割审计通过；收益不均衡 | [结果](l2_price_search_results_20261028.md) |
| 2026-10-29 fractional queue | 上轮算法 vs 当前分数 master 分离 | 六作业完成，最终割审计通过；M01/M08 终态 LB 改善，M07 提前达标主要来自 UB | [完整结果](l2_fractional_queue_results_20261029.md) |
| 2026-10-29 baseline release | 新 baseline vs 完整 CPLEX，L1 M07/M08、L2 M02/M03 | 八次均终止，七次有效且最终割审计通过；L1 CPLEX 更快，L2 M02 baseline 上下界更好，M03 完整模型内存保护停止 | [完整结果](baseline_cplex_results_20261029.md) |

## 推荐基线的选定依据

最后一轮在同二进制、相同 600 秒上限、单线程、seed43、1%目标下：

| 实例 | price+target 的 UB / LB / 秒 | fractional 的 UB / LB / 秒 |
|---|---|---|
| L2 M01 | 9704.745781 / 9609.076378 / 83.691 | 9704.745781 / 9613.852840 / 82.336 |
| L2 M07 | 5991.267163 / 5915.895806 / 599.990（限时） | 5971.607888 / 5915.239879 / 209.209 |
| L2 M08 | 3911.231233 / 3872.213378 / 520.767 | 3910.985099 / 3872.768054 / 464.316 |

三例单 seed 只支持这批结果。提前停止会影响终态 LB 比较；共同时间观测、root 松弛适用范围及新增 cut 在同一 LP 内的增量见该轮详细报告。

## 证据位置

| 批次 | 冻结目录 | 二进制 SHA-256 |
|---|---|---|
| queue EPI | `outputs/l2_queue_epi_20261028/comparison` | `ea44187730a49f2d3b2811cd6eca126a91ec2ed0edf3f793bbc94ce5ed214d71` |
| price + target | `outputs/l2_price_search_20261028/comparison` | `d81f59c5cb10d5b7507efc1324a73a943947bc4be2c579597f56b1f262d8290d` |
| fractional queue | `outputs/l2_fractional_queue_20261029/comparison` | `3a8c492e0d725ae7a32548e94b144a7cbe5e1b4c1f3d03e34b77e34bcd6747a6` |
| baseline release | `outputs/baseline_release_20261029/benchmark` | `28b1708d501855a4edd8da362bbb5eae2b743abd3176a0053b3d4d9c8612c22e` |

每批以 `experiment_manifest.json`、`state.json`、各运行 `result.json` / `native_validation.json` / `artifact_audit.json` 与最终 `analysis/CUT_AUDIT.json` 为证据。冻结后的审计修复使用单独版本并记录差异，不覆盖原程序或重新解释原终态。

本轮 baseline release 的最终审计为批次外的 `outputs/baseline_release_20261029/STREAMING_CUT_AUDIT_FINAL.json`：采用逐条读取与磁盘索引，七次有效运行通过，原八次全成功标志仍为 false。M03 完整 CPLEX 在 4 GiB 主比较及单列的 6 GiB 补充运行中均因内存停止；另行无优化导出的目录只用于 baseline 的完整域审计，不产生 CPLEX 对照成绩。

历史同模型最好可行 UB 和最强 LB 可能来自不同运行。可组合它们说明最优值所在区间，但不能回填为一次冷启动实验的 gap 或速度。M01/M07 仍有更低的历史 UB；M08 fractional 批次产生 UB 3910.985099。三例均无精确最优证明。
