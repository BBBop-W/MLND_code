# 双单支持与跨基地联合成本割：600 秒比较

新增四组：现有标准 L01/L02 × lifted/conditional Benders，K=2，各 600 秒，四组并发、每组单线程，seed=43、gap=0.005。
直接对照为 `outputs/assignment_support_benchmark_20261022` 的四组已完成结果；该批次和更早批次均保持只读。
不重新生成标准输入，不修改需求重构、物理路径可行性或运营假设。

旧参数完全继承：tight airline bounds；topology cuts（2 秒、1000 条）；assignment-support cuts（0.25 秒、2000 条、每候选最多 16 条、floor=1e-4）。
新增开关为 `--double-support-cuts --joint-base-cost-cuts`；共同分离预算 0.25 秒、新预处理预算 2 秒、数值 floor=1e-4、新割上限 2000、每候选最多 16 条。
这些参数写入新冻结清单的 `protocol.new_solver_args`；实现及数学有效性检查通过后才允许冻结并启动。
运行器禁止通过新增参数覆盖旧 `--seconds`、`--threads` 等设置。
所有新预处理、缓存构造、分离及优化均计入每组 600 秒。

单组 RSS 4 GiB，活动批次 RSS 合计 8 GiB，系统可用内存保底 2 GiB；每 0.5 秒采样，TERM 后 5 秒 KILL。
每组含独立验证的外部墙钟上限 720 秒。
资源终止不表示数学不可行。

准备阶段先重新核对上一轮源文件、输入和结果验证哈希，复制四行原始结果、四条进度日志及更早参考摘要到新批次 `reference/`。
新批次保存独立的冻结源码、二进制、输入与参考快照，全部纳入哈希清单。
准备不启动求解，且拒绝覆盖已存在批次。

待生产实现及有效性检查通过，显式记录已批准的新参数，并执行：

```bash
/opt/anaconda3/bin/python -m src.experiments.branch_joint_benchmark_20261022 prepare \
  --batch outputs/branch_joint_benchmark_20261022 \
  --new-solver-args-json '["--double-support-cuts","--joint-base-cost-cuts","--branch-joint-seconds","0.25","--branch-joint-preprocessing-seconds","2","--branch-joint-floor","0.0001","--branch-joint-cut-limit","2000","--branch-joint-max-cuts","16"]'
/opt/anaconda3/bin/python outputs/branch_joint_benchmark_20261022/run.py verify
/opt/anaconda3/bin/python outputs/branch_joint_benchmark_20261022/run.py launch
```

启动后不修改冻结实现、参数、输入或参考结果。
通过 `dispatcher.json` 的 PID 可绑定 `caffeinate -i -w PID`，调度器退出后防休眠自动结束。
实时状态见 `live_status.md`；所有组完成后自动生成 `analysis/REPORT.md` 和 `comparison.csv/JSON`。

报告重点是最终认证全局 LB 是否高于上一轮，以及是否越过实际加入主问题的连续根界 applied。
保留 raw 根界、整数主问题最好界、全部原始差值及报告容差；仅显示时抹去负零，不改 CSV 原值。
同时报告本组 UB/gap、总时间、双单和联合割分类数量、共同分离总耗时、新预处理、原割开销、SOCP 调用、模型规模、完整路径与生成列数。
未提供分类计时时不人为拆分共同耗时；数值筛选跳过是跨候选累计次数，不当作不同 OD 的数量。
不拼接不同算法的最好界计算某一组的 gap。
这是跨批次单种子比较；共享资源可能影响限时搜索，不能据此声称普遍提速。

四组均终止且独立验证通过后，生成认证下界对比曲线：

```bash
/opt/anaconda3/bin/python outputs/branch_joint_benchmark_20261022/run.py plot
```

图包含新旧四条曲线、实际采用的连续根界和独立验证后的最终点；纵轴明确采用放大成本刻度。
曲线 CSV、进度日志哈希及最终值审核保存在 `analysis/global_bound_progress.csv` 和 `figure_audit.json`。

运行器测试不调用求解器：

```bash
/opt/anaconda3/bin/python -m pytest tests/unit/test_branch_joint_benchmark.py -q
/opt/anaconda3/bin/python -m ruff check src/experiments/branch_joint_benchmark_20261022.py tests/unit/test_branch_joint_benchmark.py
```
