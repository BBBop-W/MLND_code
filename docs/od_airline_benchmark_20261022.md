# OD 需求航线割与两种 Benders 的固定规模比较

运行器：`src/experiments/od_airline_benchmark_20261022.py`。
现有 `outputs/tsl_scale_ladder/instances/L02` 与 `examples/tsl_2l20h50od` 使用相同标准 manifest，SHA256 为 `6b9e962b7d03ced667ccd90e9703fd63046c62334d826f91d165e10090298a31`；本轮直接复制 L02，不生成或修改数据。

8 组设置为 lifted/conditional × OD 航线割关闭/开启 × K=1/2。
所有组使用同一冻结二进制，保留上一轮 branch_joint 的 tight-airline、topology、assignment-support、double-support、joint-base-cost 开关和数值设置；新割开启组只追加 `--od-airline-bounds`。
每组求解预算 600 秒、单线程、3 组并行、种子 43、相对 gap 目标 0.005。
所有预处理、构造、分离和优化计入求解预算；独立验证另列。

在数学有效性检查和新实现回归通过后，使用通过检查的可执行文件准备并核验，再启动：

```bash
/opt/anaconda3/bin/python -m src.experiments.od_airline_benchmark_20261022 prepare \
  --binary /absolute/path/to/tested/mlnd_cpp
/opt/anaconda3/bin/python outputs/od_airline_benchmark_20261022/run.py verify
/opt/anaconda3/bin/python outputs/od_airline_benchmark_20261022/run.py launch
```

`prepare` 只复制与冻结，拒绝覆盖现有目录；准备完成不代表已启动。
每组保存实际命令、源码和输入哈希、逐路径流量、运营结果、独立验证、资源采样及全部失败信息。
单组 RSS 上限 4 GiB、批次活动组 RSS 8 GiB、系统可用内存下限 2 GiB，采样周期 0.5 秒，终止宽限 5 秒；每组含验证外部墙钟上限 720 秒。
资源超限不表示模型数学不可行。

运行器复用现有资源守护与独立验证，不改旧批次或旧运行器。
新的适配层按每组 K 验证案例、需求表示与实际新割开关；证书文件哈希或身份不匹配时不展示 UB/LB。
报告独立保存求解状态、构造/定价完整性、可行性、全局界与 gap，不拼接不同组的最好界。

新割对战略整数设计有效，强化后的根界可能超过原模型连续松弛值。
原连续可行目标不被当作强化后根节点的可行上界；两类根节点证据分别记录。

全部结束后自动生成 `analysis/REPORT.md`，包括新旧方法和同方法割开关两种比较。
`comparison.csv/JSON` 保存全部诊断，`paired_changes.csv` 仅计算相同 K、相同方法内的开启−关闭差值，`stage_times.csv` 是不重叠的阶段时间和占比，`cut_records.csv` 保存逐割记录。
根节点 CG 和启发式 CG 等嵌套计时另在明细中保留，不重复计入总耗时。

所有组完成且独立验证通过后绘图：

```bash
/opt/anaconda3/bin/python outputs/od_airline_benchmark_20261022/run.py plot
```

输出全局认证下界轨迹和最终 gap 的 PNG/PDF，以及轨迹 CSV 和图源哈希审核。
这是单种子、有限预算、共享计算资源比较；区分下界改善、可行上界改善及搜索轨迹变化，不据一次测试声称普遍提速。

无求解器测试：

```bash
/opt/anaconda3/bin/python -m pytest tests/unit/test_od_airline_benchmark.py -q
/opt/anaconda3/bin/python -m ruff check src/experiments/od_airline_benchmark_20261022.py tests/unit/test_od_airline_benchmark.py
```
