# 真实地图统一任务模型批量实验

入口：`src/experiments/unified_benchmark_20261027.py`。它复用已有的标准数据转换、资源监护、冻结批次、原生结果校验和运营导出，不另建画图或指标计算管线。

## 实验范围

- 美团 TSL、南山各 10 张地图；每张地图分别取 L=1、L=2，共 40 个物理实例。
- L=1 使用 10 个柜点、25 个 OD；L=2 使用 20 个柜点、50 个 OD。
- 每个实例比较单单/双单，以及 CPLEX 全路径/CPLEX Benders，共 160 个任务。
- 两种任务模式都使用 `--demand-formulation preprocessed`；底层 `--k 1/2` 只控制广义订单中是否包含双单。
- 默认保留 aggregate 流量策略，未启用 first-leg 限制。
- 每个任务包含读取、路径预处理、模型构造和求解的预算为 3600 秒，gap 为 1%，单线程，并发 4 个。
- 独立校验另记时间；外部墙钟默认 3720 秒。默认进程树 RSS 上限 4 GiB，批次合计上限 12 GiB，系统可用内存保底 2 GiB；均可显式设置。

## 地图及输入来源

`generate` 从两个真实原始 CSV 读取订单，保留对应数据源的全部历史日期，需求时段为 12–13 点、候选站点时段为 8–24 点，不缩放需求。每个订单的取餐、送达坐标都必须在地图窗口内。

每组首张地图保留历史中心。TSL 的 `outputs/tsl_scale_ladder/instances/L01` 和 `examples/tsl_2l20h50od` 原样复制；原有南山地图为 2L/10H/50OD，新组保留其地图中心及大小，但按本实验的 20H 规则重新选点，这一差异写入 provenance。

其余地图中心来自真实午间取餐位置的 0.5 km 网格，要求较小的 L=1 窗口至少包含 1000 条真实午间订单和 50 个不同 OD 坐标。由历史中心开始，按最远点规则依次选择，以实际订单数及坐标打破平局。每个数据集的两种尺寸使用相同中心；地图之间允许重叠，中心不重复。选图过程不读取任何求解结果。

每张图记录中心、尺寸、原始数据 SHA256、日期与时段、原始订单到聚合 OD 的完整归属、校准参数、需求守恒校验和标准输入文件 SHA256。地图集位于 `outputs/unified_maps_20261027`。`preparation_manifest.json` 可直接作为后续地图展示和样本统计的数据源。

## 准备与启动

```bash
/opt/anaconda3/bin/python -m src.experiments.unified_benchmark_20261027 generate

# 只冻结，不启动求解。最终公共/算法专用 flags 由本轮有效性验证决定。
/opt/anaconda3/bin/python -m src.experiments.unified_benchmark_20261027 prepare \
  --binary cpp/build-unified/mlnd_cpp \
  --seconds 3600 --gap .01 --threads 1 --parallel 4

/opt/anaconda3/bin/python -m src.experiments.unified_benchmark_20261027 verify
/opt/anaconda3/bin/python -m src.experiments.unified_benchmark_20261027 launch
```

`--common-args-json '["--shared-rho-joint","--queue-perspective","--tight-airline-bounds"]'` 指定两种方法共有的强化；`--algorithm-args-json '["--od-airline-bounds","--topology-cuts"]'` 指定算法加速。OD 航线界用于算法的 resource LP，不属于全路径模型的矩阵强化。最终命令逐任务保存在 `job.json`，不能只凭文档默认值判定实际使用的 flags。

`prepare` 拒绝覆盖已有批次，冻结代码、二进制、输入和逐原始订单 provenance，并保存当前 Git 提交、工作区状态、运行库版本及文件哈希。`launch` 启动脱离当前终端的协调进程；每次运行及报告之前均校验冻结内容。切换工作分支或修改工作区不会改变已冻结批次。

## 结果与恢复

- `dispatcher.json`：协调进程身份、启动时间和日志。
- `state.json`：全部任务和历次 attempt 状态。
- `live_status.md/csv/json`：运行期间的轻量摘要；不读取大体积解文件。
- `analysis/comparison.csv/json`：逐任务 UB、具有全局作用域的 LB、gap、阶段耗时、模型变量/线性行/锥数、完整路径数、生成列数、资源峰值、校验和运营摘要。
- `runs/<instance>/<single|double>/<method>/attempts/<number>`：求解日志、结果、设计、路径流、检查点、资源监护与校验结果；`operations/` 保留订单、路径、基地、柜子、航线和成本明细。
- `analysis/REPORT.md`：批次运行完成后的概览，缺失和失败不被隐藏。

UB 仅在独立校验以及结果、输入、二进制身份核对通过后展示。全路径数只在同一物理输入、同一任务模式之间引用。资源上限或构造超限表示当前运行配置无法完成，不表示数学不可行。带可行解的限时终止不表示证明最优。

查看状态或重新汇总不会启动求解：

```bash
/opt/anaconda3/bin/python outputs/unified_benchmark_20261027/run.py status
/opt/anaconda3/bin/python outputs/unified_benchmark_20261027/run.py report
```

正常中断后可用冻结入口的 `run --retry-failed` 恢复，已完成任务不会重复运行，重试保留原 attempt。若进程异常退出留下 `running` 状态，协调器拒绝直接重复启动，需先核对原进程是否确已结束并修复状态，避免并发重复执行。

停止整个后台批次时使用冻结入口：

```bash
/opt/anaconda3/bin/python outputs/unified_benchmark_20261027/run.py stop
/opt/anaconda3/bin/python outputs/unified_benchmark_20261027/run.py status
```

`stop` 从 `dispatcher.json` 读取 PID 与创建时间，并核对进程命令包含本批次的 `run.py`，然后向该协调器发送 SIGTERM。协调器取消运行任务并终止各自监护的进程组，保留所有已落盘文件；未启动任务仍为 pending。停止请求另存 `stop_request.json`。这是一条停止请求，返回后可用 status 检查任务是否已完成退出。不要只删除锁文件，也不要按通用进程名杀死所有 CPLEX/Python 进程。

测试入口：

```bash
/opt/anaconda3/bin/python -m pytest -q tests/unit/test_unified_benchmark_20261027.py
```
