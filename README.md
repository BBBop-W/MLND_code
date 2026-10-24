# MLND consecutive delivery network

本项目的唯一模型标识为 `consecutive`，以本次附件为基础。当前新场景默认按完整双单路径可行性筛选兼容系数，并采用用户要求的聚合任务流松弛；保留附件逐首段约束用于对照。守恒任务重构、共同续单窗口、均匀续单时间、逐单均值送达时限和 Poisson 柜格预约负荷共用。

K=1 与 K=2 共用同一模型。K=1 将有效窗口置零。旧换电站模型、旧 continuation 概率能量模型和第二单转骑手分支已移除。

## 安装与检查

Python 3.11+；Python 优化入口需要可用的 Gurobi 许可证。原生 C++ 入口使用 CPLEX，读取、求解和输出不依赖 Python。

```sh
python -m pip install -e '.[test]'
python main.py methods
python main.py inspect
python main.py plan --seconds 60
python -m pytest
python -m ruff check src tests main.py
```

`inspect` 和 `plan` 不构造优化模型、不写输入文件；构造新场景时按完整任务的物理可行性筛选兼容项，直接飞行网络使用精确存在性检查，其余路线域完整枚举，回放已保存输入则不重筛。当前默认配置读取已有 TSL 2L/10H/50OD 市场，每个柜点最多 5 个 8 格柜子（启用容量为 8/16/24/32/40 格）。后续参数调整只对照 TSL 的 K=1/K=2，使用 `compare --method full_path`。当前默认参数并不表示已经实证校准了客户选择、配对概率、柜格数或离柜时间；来源见 `configs/README.md`。

## 运行与复查

```sh
python main.py run --method benders_cg --seconds 120 --out outputs/new_run
python main.py validate outputs/new_run
python main.py report outputs/new_run
python main.py status outputs/new_run
python main.py stop outputs/new_run

# 在同一市场和场景上比较连续/非连续服务
python main.py compare --input outputs/new_run --seconds 120 --out outputs/comparison

# 比较已经保存的多算法实验，不重新求解
python main.py compare-report outputs/benchmark/runs --out outputs/benchmark/comparison

# 展开不同真实数据集的 K=1/K=2 运营指标与科学图表，不重新求解
python main.py study-report outputs/consecutive_real_comparison_20261015
```

输出目录必须是新的或空的，避免覆盖已有实验。`--input` 不带场景配置时精确重放新模型输入。`--max-orders 1` 仅改变服务模式。时间、线程、gap 等求解设置可通过命令行覆盖。

`--flow-policy aggregate` 保留各类重构任务总量、允许不同任务组合选择不同首段；`--flow-policy prefix` 保留附件逐首段比例。该选项不重算需求数组或兼容矩阵。旧快照缺少策略字段时按 `prefix` 回放。新场景的 `compatibility.require_feasible_path=true` 在优化前剔除没有完整双单路径的兼容项，关闭它只用于明确的对照实验。详细公式和信息时序边界见 `docs/model.md`。

Python 入口中，`full_path` 是小实例完整路径 MISOCP 基准；`benders_cg` 使用固定矩阵 SOCP 子问题、增量路径列生成和全局 Benders 割，默认启用 Pareto 强化割、列复用、骑手成本下界、初始设计、主问题 warm start、固定流清理及单 OD 邻域搜索。`benders_cg_plain` 关闭这七项可选策略；两者共用同一数学模型和增量加列实现。每项可用 `--enable-acceleration` / `--disable-acceleration` 独立控制，定义、认证与边界见 [算法说明](docs/algorithm.md)。

算法实验固定使用 [TSL 2L/20H/50OD 基准](data/benchmarks/tsl_2l20h50od/README.md)：

```sh
python data/benchmarks/tsl_2l20h50od/verify.py
python main.py compare --input data/benchmarks/tsl_2l20h50od \
  --method benders_cg --seconds 600 --threads 1 --seed 43 --gap 0.005 \
  --out outputs/NEW_TSL_20H_ALGORITHM
```

这一入口只回放冻结输入并切换 K；不要同时传完整物理 `--config`，以免重建参数和 κ。默认场景仍是原来的 10H 参数调整入口，20H 算法基准与其分开。原始订单和论文不参与算法修改。

Python 参考实现的定价在完整有限目录上扫描；C++ 使用精确服务阶段标号 CG，仅预处理首段/分支支持，完整路径按需动态生成；贪心初始列、证书与输出定义见 [隐式 CG 框架](docs/implicit_cg.md)。K=1 删除零续单窗口的冗余变量；aggregate 策略下，同一基地对和第一 OD 的需求递推变量按组共享，保留逐首段实际流量关联。这两项精简与固定整数设计下的原运营问题等价，证明和松弛边界见算法说明。Python 的目录、时间或构模规模超限会返回有效的全骑手基线及明确停止原因，不会把截断目录当成完整模型。报告需同时读取 incumbent、bound、gap、pricing_complete、path_scope 和 termination_reason。

## C++ / CPLEX 标准实例入口

固定 TSL 20H 的可移植 CSV/JSON 实例位于 `examples/tsl_2l20h50od/`。完整模型与 Benders 共用原生模型构造和结果格式：

```sh
make -C cpp -j2
cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 1 \
  --method full_path --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/NEW_CPP_K1
python -m src.experiments.native validate outputs/NEW_CPP_K1 \
  --instance examples/tsl_2l20h50od
```

`--k 2` 切换为双单任务；`--method benders_cg` 使用原生 Benders、规范列生成、解析成本与资源割、少量运营变量松弛、根节点割、分支覆盖种子和受限列联合优化启发式；数学依据与预算见 [主问题加强](docs/master_strengthening.md)。构建要求、固定设计互验、开关及全部输出字段见 [C++ 使用说明](cpp/README.md)。两求解器在限时 MIP 搜索中可能得到不同可行解；模型对齐通过输入、路径、系数矩阵、固定设计及独立运营约束逐层检查。

C++ 定价按有序基地对扩展标号，保留 AA 和跨基地任务，使用经过完整后续扩展验证的支配规则。每轮受限 SOCP、对偶间隙和完整定价均有 `cg_trace`；只有全部认证通过才产生新 Benders 割。`--cg-max-columns` 限制加列批量，不截断精确定价；`--fixed-design ... --method benders_cg` 用于同设计全列对照。算法依据见 [多基地定价研究](docs/label_pricing_research.md)。

## 从原始订单构造新实例

1L/10H/25OD 至 10L/100H/250OD 的固定密度 TSL 阶梯实验，使用独立的生成、冻结、三任务调度和完整验收入口，见 [五小时阶梯实验说明](docs/scale_ladder.md)。共 60 组，每组 5 小时、1 线程；准备和预检不求解，冻结入口不带参数只显示状态。

```sh
python main.py prepare --source nanshan --bases 1 --lockers 3 --ods 4 \
  --half-size-km 2 --out outputs/prepared/nanshan_small
python main.py run --input outputs/prepared/nanshan_small \
  --method full_path --seconds 60 --out outputs/nanshan_small_full
```

基地数、柜点数、OD 数、日期和时段分别设置，不再固定比例。原订单按联合 OD 聚类，每条订单计入一次；原始分段运价先逐订单计算再聚合。时间记录不完整的订单仍进入需求量，时间估计方法和假设单独记录。`prepare` 不允许写入原始数据、论文或已有实例目录。

## 代码结构

```text
main.py                   命令行入口
configs/                  当前模型默认场景
cpp/                      原生 C++ / CPLEX 完整模型、Benders、标准实例读写
examples/tsl_2l20h50od/    冻结 TSL 20H 的 CSV/JSON 标准实例
src/
  cli.py, config.py       命令与配置检查
  data/                   原订单、候选点、实例、明确场景
  model/                  数据类型、需求、物理路径、固定资源
  solvers/                MISOCP、SOCP、对偶、CG、Benders
  experiments/            运行、进度、保存、状态
  operations/             独立物理重算、守恒核验、表格导出
  reporting/              成本、机队、工作量、时间、网络图
tests/
  unit/                   数学、求解证书、数据和篡改检查
  integration/            求解到保存、回读、核验和绘图
docs/
  model.md                模型口径与公式映射
  architecture.md         接口、结果和验证规则
```

原始数据、`data/`、`MLND/` 和参考论文保持原位。旧 `mlnd_path/` 中发现的四组论文/LaTeX 文件也原位保留；该目录内旧求解器和实验脚本已删除，论文附属绘图/核验工具随论文保留。历史实验结果目录没有被用作新模型实验结论。

每次运行保存输入 JSON/NPZ、解数组、带物理节点的路径、独立核验表、报告和源码哈希。源码不再逐次复制。图形导出 PNG/PDF，地图默认使用本地公里坐标，不访问网络；显式申请缓存底图需要投影与坐标系来源，缺失时报告错误。

运行进度同时保存为最新状态 `progress.json` 和带实测时间的完整历史 `progress.jsonl`。算法比较要求物理输入和路径范围一致，K=1/K=2 分别解释。历史快照只在读取时统一模型名称，物理参数和原文件不变。

2026-10-15 的 2L/10H/50OD 南山实验入口为 `outputs/consecutive_benchmark_20261015/`，包含切片比较、逐订单聚类映射、场景配置、可重放命令和独立核验结果。

当前聚合任务流下的南山/美团 TSL 同规模 K=1/K=2 对照见 `outputs/consecutive_real_comparison_20261015/report/REPORT.md`。该实验使用两源历史 region_1 的相同面积切片、真实未缩放需求与统一处理程序，保留逐基地、柜点、OD、航段、任务组合、成本和无人机忙碌时间明细，图形同时导出 PNG/PDF。输入审计、独立结果审计、求解源码哈希与后处理源码哈希分别保存。
