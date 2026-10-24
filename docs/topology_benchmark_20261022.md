# 拓扑割与航线界强化的小规模比较

运行器 `src/experiments/topology_benchmark_20261022.py` 复用已测试的条件 Benders 运行器、进程组资源保护、独立验证和实时状态功能，不修改原运行器或原输入。

使用既有标准 L01（1 基地、10 柜、25 OD）和 L02（2 基地、20 柜、50 OD），K=2。
每个实例比较五个变体，共 10 组：

| 变体 | 需求表示 | 新强化 |
|---|---|---|
| lifted_baseline | 原 lifted Benders | 两项均关闭，沿用旧默认命令 |
| conditional_baseline | 条件 Benders | 两项均关闭，沿用旧默认命令 |
| lifted_strengthened | 原 lifted Benders | tight airline bounds + topology cuts |
| conditional_strengthened | 条件 Benders | tight airline bounds + topology cuts |
| full_cplex_reference | CPLEX 完整路径 | 原默认设置，提供参考 |

全部组使用同一份冻结的新二进制；强化组额外设置 `--tight-airline-bounds --topology-cuts --topology-seconds 2 --topology-cut-limit 1000`。
旧方法和新方法原有的其他默认加速保持一致。
本轮检验两项强化一起使用的效果，不单独识别每项强化的贡献。

每组最多 **600 秒求解预算**，不是整个批次 10 分钟；预算涵盖输入、路径准备、构造和优化。
每组单线程、4 组并行、gap=0.005、seed=43。
含验证的单组外部墙钟上限为 720 秒；单组 RSS 上限 4 GiB、批次活动组 RSS 合计 8 GiB、系统可用内存至少 2 GiB。
并行运行存在资源竞争；原始墙钟、求解阶段时间、资源停止原因均单独报告。

通过模型与新二进制正确性检查之后，冻结而不启动：

```bash
/opt/anaconda3/bin/python -m src.experiments.topology_benchmark_20261022 prepare \
  --batch outputs/topology_benchmark_20261022 --levels 1 2 \
  --seconds 600 --parallel 4 --threads 1 --gap .005 --seed 43 \
  --hard-wall-seconds 720
```

若新构建位于其他位置，显式传入 `--binary /absolute/path/mlnd_cpp`。
冻结后用该批次的入口校验并后台启动：

```bash
/opt/anaconda3/bin/python outputs/topology_benchmark_20261022/run.py verify
/opt/anaconda3/bin/python outputs/topology_benchmark_20261022/run.py launch
```

后台 PID 记录于 `dispatcher.json`，可用 `caffeinate -i -w PID` 防止运行期间空闲休眠。
查看 `live_status.md` 可见已完成组的 UB、LB、gap、终止原因。
全部完成后自动生成 `analysis/REPORT.md` 与 `analysis/comparison.csv/JSON`。
报告也可通过冻结入口 `run.py report` 重新生成；未完成批次明确标记，不形成性能结论。

报告包括总成本 UB、全局 LB、gap、分阶段耗时、主问题与整数子问题规模、完整路径与已生成列数、拓扑割尝试/数量/跳过整数子问题次数、连续根界、整数主问题界、全骑手候选比例以及总成本相对全骑手基准的比例。
连续根界仅在 `bound_valid=true` 时列出；整数主问题界不替代最终全局 LB。
全骑手候选比例的分母是实际 `integer_cg_calls`；被拓扑割跳过的候选不计入此分母。
所有可行解均经过既有独立验证与哈希核对；内存或墙钟退出不表示数学不可行。

运行器功能测试不求解性能算例、不生成新标准输入：

```bash
/opt/anaconda3/bin/python -m pytest tests/unit/test_topology_benchmark.py -q
/opt/anaconda3/bin/python -m ruff check src/experiments/topology_benchmark_20261022.py tests/unit/test_topology_benchmark.py
```
