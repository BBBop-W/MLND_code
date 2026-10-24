# K=2 条件 Benders 对比运行说明

运行器：`src/experiments/conditional_benchmark_20261022.py`。
它复用现有 `src.experiments.native` 的 Gurobi 完整路径入口、独立运营验证，以及 `scale_ladder.execute/audit_attempt` 的日志、资源采样和原始产物审计。
运行器不生成算例，不改模型参数或原始输入。

本轮选用已经存在的 `outputs/tsl_scale_ladder/instances/L01`、`L02`、`L03`，规模分别为 1L/10H/25OD、2L/20H/50OD、3L/30H/75OD。
每个算例比较四种方法：CPLEX 全路径、Gurobi 全路径、CPLEX 原有 Benders-CG 完整提升子问题、CPLEX 条件 Benders-CG 精简子问题。
K 固定为 2，原有需求重构、分配和首段比例约束均以冻结代码为准。

默认每组 180 秒总预算、单求解器线程、种子 43、目标相对 gap 0.005，顺序运行各组以减少 CPU 和内存竞争。
预算包括读取、完整路径枚举或隐式支持准备、模型构造和优化；最终输出与独立验证另列时间。
完整路径方法不设置路径数或变量数人为截断上限。
默认保留各方法支持的加速策略；两种 Benders 均使用原有每轮 512 列插入上限（这不是完整路径域上限）。

先完成当前源码的构建和正确性验证，再冻结；冻结后的文件由 SHA256 校验，不能在原批次中替换二进制或源码：

```bash
/opt/anaconda3/bin/python -m src.experiments.conditional_benchmark_20261022 prepare \
  --batch outputs/conditional_benchmark_20261022 \
  --seconds 180 --threads 1 --parallel 1 --gap 0.005 --seed 43
```

`prepare` 将现有实例逐文件复制进新批次，冻结 `src`、`configs`、`cpp` 源码和 `cpp/build/mlnd_cpp`；不启动求解器。
源码复制排除 `build*` 编译目录，只在 `frozen/bin` 中保留选定的实验可执行文件。
本机 `/opt/anaconda3/bin/python` 已能启动 Gurobi 13.0.2 环境；仓库 `.venv` 缺少 `gurobipy`，不使用该解释器。

从冻结入口执行，以免运行过程中读取后来修改的工作区源码：

```bash
/opt/anaconda3/bin/python outputs/conditional_benchmark_20261022/run.py verify
/opt/anaconda3/bin/python outputs/conditional_benchmark_20261022/run.py run
/opt/anaconda3/bin/python outputs/conditional_benchmark_20261022/run.py report
```

可以在 `run` 后用 `--jobs L01_K2_benders_conditional` 仅调度指定任务。
默认只运行 pending 任务；使用 `--retry-failed` 会创建新 attempt，不覆盖旧结果。
不同预算、加速开关或源码版本应准备新目录，不能混入同一批次。

最终报告位于 `analysis/REPORT.md`，详细数据位于 `analysis/comparison.csv` 和 `comparison.json`。
报告包括：UB、全局 LB、gap、预算内耗时、进程墙钟、各阶段时间、完整模型和 Benders 主/子问题规模、完整路径数及实际生成列数、终止状态、构造和定价完成性、独立验证、原始结果哈希。
完整路径数只从同输入、同 K 的已完成目录读取，目录不一致时报告器拒绝合并。
Benders 已生成列数不被当成完整路径数。
子问题末次和最大规模统计整数设计的运营子问题，包含启发式访问的整数设计；连续根节点模型不计入该统计。
未经独立验证或验证哈希过期的结果不进入 UB/LB 比较。

每个 attempt 保留原始 `stdout.log`、`stderr.log`、`execution.json`、资源采样、求解结果、路径流量、验证与运营表。
进程失败、建模超时、有效可行解、认证下界及间隙分别呈现；不把非零退出码简化为“无可行解”，也不把限时可行解称为最优解。

## 10 级夜间比较与资源保护

以下命令只使用已存在的 L01–L10 标准输入（1L/10H/25OD 至 10L/100H/250OD），共 40 组，不生成输入。
每组求解器总预算 7200 秒，4 组并行，每组单线程；求解后的验证继续占用本组并行槽。
本机 24 GiB 内存下，采用单组 RSS 上限 4 GiB、批次活动进程组 RSS 合计上限 8 GiB、系统可用内存下限 2 GiB。
资源保护只终止本批次拥有的进程组。

```bash
/opt/anaconda3/bin/python -m src.experiments.conditional_benchmark_20261022 prepare \
  --batch outputs/conditional_overnight_20261022 \
  --levels 1 2 3 4 5 6 7 8 9 10 \
  --seconds 7200 --threads 1 --parallel 4 --gap 0.005 --seed 43 \
  --job-rss-gib 4 --batch-rss-gib 8 --system-available-gib 2 \
  --hard-wall-seconds 7320 --resource-sample-seconds 0.5 \
  --terminate-grace-seconds 5
/opt/anaconda3/bin/python outputs/conditional_overnight_20261022/run.py launch
```

`launch` 创建脱离终端的调度进程，记录 `dispatcher.json` 中的 PID 和创建时间，并把输出保存到 `dispatcher.log`。
它不依赖发起对话继续运行；调度器锁阻止同一批次重复启动。
macOS 若需避免运行期间空闲休眠，可对返回的调度器 PID 使用 `caffeinate -i -w PID`；这不修改系统设置。

资源守护器每 0.5 秒采样一次，RSS 包括本组工作进程、守护器、求解器及留在该专用进程组的后代。
macOS 使用限定进程组的 `proc_listpgrppids`，不依赖全系统进程枚举权限；求解器退出后仍会清理同组残留后代。
当前已知求解器调用链保持在同组；自行建立新会话脱离该组的其他程序不在这项保护的适用范围内。
采样失败或活动记录过期时停止对应工作，不把无法读取的 RSS 当成零。
总量或系统内存保护用互斥租约选择占用最大的一个活动组，避免多个守护器因同一次压力同时终止多组。
检测到超限先发送 SIGTERM，宽限 5 秒后发送 SIGKILL；7320 秒外部墙钟覆盖求解和验证，TERM 在到达上限前启动，采样和操作系统调度可能带来少量延迟。

记录区分 `construction_memory_limit`、`solve_memory_limit`、`validation_memory_limit`，并单列 `per_job_rss_limit`、`batch_rss_limit` 或 `system_available_memory_floor`。
这些表示在设定资源预算下无法完成，不表示模型数学不可行。
没有最终 `result.json` 或文件写入被中断时，停止原因、阶段、采样峰值、日志和已有 checkpoint 仍然保留；只有通过独立验证和哈希核对的 checkpoint 才用于报告界与可行解。

大文件验证和最终报告也在资源守护的独立进程中运行，调度器不直接加载巨大的路径流量或结果文件。
每组结束后由验证进程保存轻量 `report_row.json`，调度器据此原子更新批次根目录的 `live_status.md`、`live_status.csv` 和 `live_status.json`。
Markdown 状态表使用北京时间和百分比 gap；原始 JSON、日志时间保留 UTC。
因此批次未全部完成时也可查看已完成各组的界、间隙、停止原因和峰值。
全部任务结束后自动生成完整报告；报告若因资源或文件错误失败，保留实时状态、原始数据及 `report_failure.json`，不会影响其他任务的调度。
