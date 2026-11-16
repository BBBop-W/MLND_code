# 固定广义订单预处理：L01 三模型比较

本实验用已冻结的 `outputs/tsl_scale_ladder/instances/L01`，比较当前单单、原双单，以及先预处理广义订单再规划的新双单模型。
输入为真实 TSL 订单聚合的 1 个基地候选、10 个柜点、25 个 OD；6904 条观测订单对应八天每天 12–13 点共 8 小时，需求量为 863 单/小时。
续单窗口为 30 秒，现有兼容矩阵有 336 个非零项，全部为 0/1。
实验不重新聚类、不生成新算例，也不修改原始输入系数。

| 模式 | 命令参数 | 需求与分配 |
|---|---|---|
| 单单 | `--k 1 --demand-formulation lifted` | 当前 OD 到出发基地分配 |
| 原双单 | `--k 2 --demand-formulation lifted` | 原分配相关需求重构，OD 到基地对分配 |
| 预处理双单 | `--k 2 --demand-formulation preprocessed` | 需求在优化前重构，同一首单 OD 的所有任务共享出发基地 |

三组采用同一个冻结 CPLEX 可执行文件、完整路径、600 秒总预算、相对 gap `1e-6`、单线程及种子 43，顺序运行。
总预算包含输入、完整路径枚举、需求预处理、构模和求解；输出及独立验证另计。
原完整目录的 K=1 和 K=2 路径数分别为 134 和 6075，实际运行仍重新核对目录完成状态和数量。
所有物理可行路径均进入完整模型，零需求分支是否被 presolve 消去由求解器处理；不把删减目录当成全路径基准。

预处理使用未被此前配对占用的残余率 `a`。
对不同 OD 的有序对 `(r,s)`，采用 `q=1-exp(-a_s delta)`、`gamma=min(a_r,a_s)q`，同时从两个残余率扣除 `gamma`。
自配对 `(r,r)` 使用 `gamma=a_r q/2`，从残余率扣除 `2 gamma`。
结束后的残余率为单单需求；每个 OD 的单单加上作为首单、续单的流量等于原始需求，自配对计两次。

候选对先按单单和双单最便宜可行路径的外生成本差排序，保持确定性的 OD 编号并列顺序；只处理正代理节省的组合。
代理成本采用既有路径运营成本，不含机队、设施固定费、空闲库存或柜格拥挤成本，因此它不是完整网络收益的证明。
候选对还须兼容且存在物理可行路径；增加正需求分支时检查同一首单的全部分支仍至少有一个共同可行的出发基地。
上述排序和匹配率是明确的预处理启发式；残余流按泊松率继续处理属于近似，流量守恒是代数保证，不证明随机在线匹配可实现。
排序、每次扣除、支持集合、单单及双单率保存到结果的 `preprocessed_demand` 中，便于逐项独立复核。

新模型中，无人机单单流量为 `nu_r alpha_rl rho_l`，双单流量为 `nu_rs alpha_rl rho_l`；路径可返回任一可行基地。
每条原始 OD 的直接骑手流量等于原始需求减去所有路径中的实际服务次数，双单任务计两个订单。
未被无人机执行的双单任务两单均进入直接骑手需求，不额外隐含拆为两条无人机单单任务。
原有机队、基地平衡、路径均值时间和时效、柜格占用及容量约束沿用公共构造。

L01 只有一个基地，因此起点分配和基地对分配的返回基地自由度在这个算例中一致。
仍需区分另一个变化：新模型中某 OD 未作为首单分配时，也可能作为另一 OD 的续单接受服务；原双单版本要求该 OD 本身进入相同基地需求池。
需求率固定在选址之前也是新的建模选择。
所以本实验比较三种运营规划模型，不将收益全部归因于连续接单次数或同模型算法加速。

实验入口复用现有 `scale_ladder.execute` 的独立进程组、日志、外部超时与 child `wait4` 资源统计，并复用 `native validate` 的物理和运营重算。
不生成专用绘图代码。
变量、行和锥数量指原始完整模型，不是 presolve 后规模。
求解进程组外部时限为 660 秒，验证为 120 秒；本小算例未设置内存硬限制，进程树采样不可用时保留缺失标记及 child `wait4` 峰值。

在代码构建及针对性验证完成后冻结；`prepare` 不启动求解器。
旧批次不覆盖，运行时使用冻结副本，并逐文件验证原输入仍与清单一致。

```bash
make -C cpp -j2
/opt/anaconda3/bin/python -m src.experiments.generalized_demand_benchmark prepare \
  --batch outputs/generalized_demand_L01_20261026 \
  --instance outputs/tsl_scale_ladder/instances/L01
/opt/anaconda3/bin/python outputs/generalized_demand_L01_20261026/run.py verify
/opt/anaconda3/bin/python outputs/generalized_demand_L01_20261026/run.py run
/opt/anaconda3/bin/python outputs/generalized_demand_L01_20261026/run.py report
```

`run --jobs single legacy_double` 可先运行指定的 pending 组；完整命令会跳过已经记录的非 pending 组。
失败或中断的组不会被自动覆盖，如需重新试验应建立新批次。

最终结果位于 `analysis/REPORT.md`、`comparison.csv` 和 `comparison.json`。
报告展示 UB、认证全局 LB、gap、终止状态、可行性、预算内及进程墙钟时间、构模规模、完整路径数、机队、柜格和实际单单/双单服务量。
新模型额外报告正需求双单类型数、重构任务率和逐 OD 最大守恒误差。
仅通过独立验证且结果、解和路径文件哈希相符的结果进入成本比较；限时解与最优性证书分开陈述。
输入、源码与可执行文件的哈希保存在 `experiment_manifest.json`。

## 固定重构需求，比较是否要求首程一致

同一运行器支持一个单独的两组实验：两组均为预处理双单，一组保留首单 OD 与出发基地层面的汇总分支比例，另一组增加 `--first-leg-consistency`。
后一组要求同一实际首程的单单与各双单流量按预先重构的分支比例出现；返回基地仍可由后续路径选择。
预处理需求率、配对排序、兼容矩阵、完整物理路径目录和所有成本均相同。
该实验不重新选择配对来补救首程支持不足，也不删除不便服务的正需求分支。

若某首单没有一个可同时支持全部正需求分支的首程，那么首程一致约束可能禁止该首单的正任务流量。
报告保留这一结果，并给出首程数、支持所有分支的首程数，以及有/无完整首程支持的首单 OD 数；未服务订单仍计入直接骑手余量。
首程一致版本仅增加约束，因此其理论最优成本不低于汇总比例版本。
比较限时解时仍须使用两组上下界，不能仅凭两个可行目标判断最优成本差。

```bash
/opt/anaconda3/bin/python -m src.experiments.generalized_demand_benchmark prepare \
  --first-leg-comparison \
  --batch outputs/generalized_first_leg_L01_20261026 \
  --instance outputs/tsl_scale_ladder/instances/L01 \
  --reference-batch outputs/generalized_demand_L01_20261026
/opt/anaconda3/bin/python outputs/generalized_first_leg_L01_20261026/run.py verify
/opt/anaconda3/bin/python outputs/generalized_first_leg_L01_20261026/run.py run
```

新的两组再次使用同一冻结二进制，预算仍为每组 600 秒、gap `1e-6`、seed 43、单线程、顺序运行。
单单和原双单只读取前一批次已经验证的结果作为背景参考，不重新求解，不混入本轮同二进制耗时比较。
运行器对两组重构率、首单率、配对顺序和共同出发基地支持逐项比较，发现不一致就拒绝输出受控比较结论。
旧三组实验入口及默认参数不变；新实验写入独立批次，不改动原结果。
