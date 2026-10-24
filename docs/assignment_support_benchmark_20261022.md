# 分配与路径支持成本割：600 秒对照

本轮只新增四组：既有 L01（1 基地、10 柜、25 OD）与 L02（2 基地、20 柜、50 OD），各运行 lifted 和 conditional Benders，K=2。
沿用上一轮 strengthened 的 `--tight-airline-bounds --topology-cuts --topology-seconds 2 --topology-cut-limit 1000`，仅增加分配与路径支持成本割。
每组 600 秒、四组并发、单线程、seed=43、gap=0.005。
新分离、缓存构造和后续优化都计入 600 秒；不修改需求重构、运营假设、输入或其他加速。

对照直接读取已完成的 `outputs/topology_benchmark_20261022`，包括原 baseline、strengthened 和完整路径参考。
该目录保持只读。
准备新批次时重新检查旧源码/输入冻结哈希与独立验证记录，复制参考行和进度日志到新批次的 `reference/`，加入新哈希清单。
新旧比较是跨批次单种子比较，不把并发资源竞争造成的差异当作算法的普遍优势。

主要指标为最终认证全局 LB、相对上一轮 strengthened 的 LB 改变、以及最终 LB 是否超过本轮认证连续根界。
后者同时给原始差值，判断容差为 `1e-7 * max(1, abs(continuous_root))`，避免数值舍入被误计为提升。
同时报告本组 UB/gap、整数主问题最好界、割数量/分离开销/跳过 SOCP 数、模型与路径规模及终止状态。
不将不同方法的最好 UB/LB 拼接为任何单组的 gap。

单组 RSS 上限 4 GiB，活动批次合计 8 GiB，系统可用内存保底 2 GiB；每 0.5 秒采样，超限 TERM 后 5 秒 KILL。
每组含独立验证的外部墙钟上限 720 秒。
资源停止不表示数学不可行。

只有在数学有效性检查和实现回归通过后，才准备/冻结并启动：

```bash
/opt/anaconda3/bin/python -m src.experiments.assignment_support_benchmark_20261022 prepare \
  --batch outputs/assignment_support_benchmark_20261022 \
  --assignment-support-seconds 0.25 --assignment-support-cut-limit 2000 \
  --assignment-support-max-cuts 16 --assignment-support-floor 0.0001
/opt/anaconda3/bin/python outputs/assignment_support_benchmark_20261022/run.py verify
/opt/anaconda3/bin/python outputs/assignment_support_benchmark_20261022/run.py launch
```

四个数值开关与最终审定的求解器接口一致；正式运行显式记录这些值。
未在准备命令中指定时，运行器只添加 `--assignment-support-cuts` 并使用已冻结二进制的默认值。
启动后不修改冻结源码、设置或参考数据。
后台调度器 PID 保存在 `dispatcher.json`；实时状态见 `live_status.md`。

四组全部结束后自动生成 `analysis/REPORT.md` 与 `comparison.csv/JSON`。
通过独立验证后可生成认证下界轨迹图：

```bash
/opt/anaconda3/bin/python outputs/assignment_support_benchmark_20261022/run.py plot
```

图仅比较旧 strengthened 与新增割，包含本轮连续根界参考；纵轴明确标作放大成本刻度。
中间曲线来自程序进度日志，末端标记来自独立验证后的最终结果。
图源 CSV、日志哈希和审核信息分别保存在 `analysis/global_bound_progress.csv` 与 `analysis/figure_audit.json`。

运行器测试不启动求解、不生成新标准输入：

```bash
/opt/anaconda3/bin/python -m pytest tests/unit/test_assignment_support_benchmark.py -q
/opt/anaconda3/bin/python -m ruff check src/experiments/assignment_support_benchmark_20261022.py tests/unit/test_assignment_support_benchmark.py
```
