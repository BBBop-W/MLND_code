# MLND consecutive delivery network

本项目的模型标识为 `consecutive`，研究基地、柜点、航线、订单分配和无人机运营流量的联合规划。Python / Gurobi 与 C++ / CPLEX 使用共同的物理路径、时间、柜格和闭排队网络口径。

本分支 `codex/k1-origin-assignment` 仅调整 **K=1 的订单分配**：每个 OD 最多指定一个出发基地，返回基地由运营路径选择。同一 OD 可以通过不同路径返回不同基地，所有基地仍满足无人机流量平衡。K=2 保留 OD 到出发—返回基地对的分配、既有连续接单需求重构及 `aggregate` / `prefix` 规则。公式、旧新模型的关系和兼容边界见 [K=1 起点分配说明](docs/k1_origin_assignment.md)。

| 模式 | 二元分配 | 路径的物理端点 | 对照含义 |
|---|---|---|---|
| 当前 K=1 | OD × 出发基地 | 出发基地、返回基地均保留 | 放宽同一 OD 必须返回同一基地的限制 |
| 当前 K=2 | OD × 出发基地 × 返回基地 | 出发基地、返回基地均保留 | 沿用本分支之前的连续接单模型 |
| 历史 K=1 | OD × 出发基地 × 返回基地 | 出发基地、返回基地均保留 | 用于历史复查，不能直接当作新 K=1 的同模型算法结果 |

新场景默认按完整双单路径可行性筛选兼容系数，采用聚合任务流规则，保留逐首段约束作为对照。K=1 的有效续单窗口仍为零，但当前 K=1 与 K=2 **还存在分配域的差别**；比较两者时不能把所有成本差异都归因于连续接单。旧换电站模型、旧 continuation 概率能量模型和第二单转骑手分支已移除。

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

输出目录必须是新的或空的，避免覆盖已有实验。`--input` 不带场景配置时重放保存的物理输入；本分支 `--max-orders 1` 同时采用 K=1 的起点分配域。重放输入不意味着恢复历史版本的优化模型。时间、线程、gap 等求解设置可通过命令行覆盖。

**输入格式要与入口对应。** `main.py run --input` / `compare --input` 读取 Python 快照，需要 `input_metadata.json` 和 `input_arrays.npz`，例如 `data/benchmarks/tsl_2l20h50od/`。`examples/tsl_2l20h50od/` 是原生 CSV/JSON 实例，包含 `model.json`、`sites.csv` 等文件，应交给 C++ 的 `--input` 或 `src.experiments.native` 的 `--instance`，不能直接传给 `main.py run --input`。

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

Python 参考实现的定价在完整有限目录上扫描；C++ 使用精确服务阶段标号 CG，仅预处理首段/分支支持，完整路径按需动态生成；贪心初始列、证书与输出定义见 [隐式 CG 框架](docs/implicit_cg.md)。K=1 不保留零续单窗口的冗余递推变量，并采用本分支的起点需求池。K=2 的 aggregate 策略下，同一基地对和第一 OD 的需求递推变量按组共享，保留逐首段实际流量关联；递推变量精简的证明和松弛边界见算法说明。Python 的目录、时间或构模规模超限会返回有效的全骑手基线及明确停止原因，不会把截断目录当成完整模型。报告需同时读取 incumbent、bound、gap、pricing_complete、path_scope 和 termination_reason。

## C++ / CPLEX 标准实例入口

固定 TSL 20H 的可移植 CSV/JSON 实例位于 `examples/tsl_2l20h50od/`。完整模型与 Benders 共用原生模型构造和结果格式：

```sh
make -C cpp -j2
cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 1 \
  --method full_path --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/NEW_CPP_K1
python -m src.experiments.native validate outputs/NEW_CPP_K1 \
  --instance examples/tsl_2l20h50od

# 在同一个原生 CSV/JSON 实例上运行 Gurobi 全路径基准
python -m src.experiments.native gurobi \
  --instance examples/tsl_2l20h50od --k 1 \
  --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/NEW_GUROBI_K1
```

`--k 2` 切换为双单任务；`--method benders_cg` 使用整数设计主问题、固定设计 SOCP、规范列生成、解析成本与资源割、根节点割、分支覆盖种子和受限列联合优化启发式；数学依据与预算见 [主问题加强](docs/master_strengthening.md)。构建要求、固定设计互验、开关及全部输出字段见 [C++ 使用说明](cpp/README.md)。两求解器在限时 MIP 搜索中可能得到不同可行解；模型对齐通过输入、路径、系数矩阵、固定设计及独立运营约束逐层检查。

K=1 的原生输出使用 `assignment_scope="origin"`，`assignment` 与对应乘积变量的维度为 `(R,L)`。`--assignment-support-cuts`、`--double-support-cuts`、`--joint-base-cost-cuts` 的现有证书依赖固定返回基地，因此在 K=1 明确拒绝这些选项；K=2 保持原有适用范围。其他按 OD/基地对编号编写的历史诊断，也应先检查接口，而不能直接套用历史设计文件。

Python 新结果在 `result.json` / `manifest.json` 顶层记录 `assignment_scope`；数值解数组保持在 `solution.json` / NPZ。历史 K=1 三维解可通过独立验证和运营导出按原来的 `base_pair` 语义复查，并在摘要与 OD 分配表中标识。新固定设计和 warm start 要求当前模型的数组维度，旧三维 K=1 设计会明确拒绝，不自动转换。

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

### 从哪里开始读和修改

| 工作内容 | 权威入口 | 复用方式 |
|---|---|---|
| 配置、命令行与方法选择 | `src/config.py`、`src/cli.py`、`src/solvers/methods.py` | 通过现有命令入口传参数，避免在新脚本中复制默认值 |
| 实例结构与需求重构 | `src/model/types.py`、`demand.py`、`compatibility.py` | 保留 OD、基地、柜点的 ID 顺序；K=2 重构不要在报告层重新实现 |
| 路径及时间/格口资源 | `src/model/resources.py`、`paths.py` | 使用 `build_service_path`、`rebuild_path`、`enumerate_paths`；不要另写一套飞行时间或柜格占用公式 |
| Gurobi 公共模型与增量列 | `src/solvers/formulation.py` | `add_design`、`build_model`、`path_column`、`add_path_columns` 供完整模型和 Benders 共用 |
| Gurobi 求解、认证及加强 | `src/solvers/algorithm.py`、`dual.py`、`pareto.py`、`strengthening.py` | 方法开关走 `api.py` / `methods.py`；割必须与当前分配域及完整路径域一致 |
| CPLEX 公共模型 | `cpp/src/input.cpp`、`paths.cpp`、`formulation.cpp`、`linear_model.cpp` | 原生读取、路径资源、公共稀疏矩阵与求解器接口分开；各算法复用公共矩阵 |
| CPLEX Benders 和定价 | `cpp/src/benders.cpp`、`pricing.cpp`、`dual_workspace.cpp`、`heuristics.cpp` | 复用原始/对偶证书、标号定价、列池、启发式和预算管理 |
| CPLEX 加强割及诊断 | `cpp/src/strengthening.cpp`、`topology_cuts.cpp`、`assignment_support_cuts.cpp`、`branch_joint_cuts.cpp`、`bdd*.cpp` | 生产算法与独立诊断分开；新增割先做有效性测试，再做等预算比较 |
| 运行、保存与资源限制 | `src/experiments/runner.py`、`storage.py`、`progress.py`、`resource_guard.py` | 使用现有保存协议、进度和限制；不要在分析脚本中顺带启动求解器 |
| 独立运营核验 | `src/operations/validation.py`、`export.py`、`detail.py` | `build_operations` 重建物理量与约束，`export_operations` 导出表，`enrich_operations` 生成细分指标 |
| 原生结果互验 | `src/experiments/native.py` | 原生 JSON/CSV 结果先适配和验证，再进入报告；`gurobi` 子命令会调用 Gurobi，`validate` 不求解 |
| 报告与绘图 | `src/reporting/`，见下一节 | 先选已有输入格式和报告入口，再扩展已有函数 |

`cpp/README.md` 给出完整构建、算法开关和原生输出说明；`docs/architecture.md` 说明数据与验证接口；`docs/model.md` 记录基础模型，当前 K=1 的差异以 `docs/k1_origin_assignment.md` 为准。带日期的实验文档、`docs/bdd*.md` 和诊断脚本分别对应其冻结版本，不能仅因仍在仓库中就认定为当前默认算法。

## 报告与绘图：优先复用已有入口

画图前先识别结果格式。Python 保存目录、原生 C++ 目录和批量实验目录的布局不同，不能只把文件名改成相同就交给同一个读取器。

| 任务与输入 | 已有命令 / 函数 | 主要输出 |
|---|---|---|
| 一个 Python 保存运行 | `python main.py report RUN --out REPORT`；`src.reporting.report.run` | 独立核验、运营明细、成本/机队/负荷图、网络图和 `REPORT.md` |
| 同市场多个 Python 运行 | `python main.py compare-report RUN_PARENT --out REPORT`；`produce_comparison` | 方法表、上下界/收敛图、运营比较、网络图 |
| 多数据集 K=1/K=2 运营研究 | `python main.py study-report STUDY_ROOT --out REPORT`；`produce_study` | 多数据集指标、逐基地/柜点/OD 明细及解释性报告；默认读取 `STUDY_ROOT/runs/{dataset}_K1`、`{dataset}_K2` |
| 既有原生三方法比较目录 | `python -m src.reporting.native EXPERIMENT_ROOT --out REPORT`；`build_native_report` | `full_cplex`、`full_gurobi`、`benders_cplex` 下 K1/K2 的核验后比较；目录标签由模块 `RUNS` 定义 |
| 规模阶梯批次 | `python -m src.reporting.scale_ladder BATCH`；`build_scale_report` | 从批次 manifest、任务状态、结果和核验记录生成 `BATCH/analysis` |
| 结构加强 / BDD 既有实验 | `src/reporting/structural_comparison.py`、`bdd.py` 与对应 `src/experiments/` 的 `report` 入口 | 使用该实验自身的协议和目录结构；勿套用不匹配的通用读取器 |

以上报告模块只读取保存结果，不重新优化。原生单次结果先执行 `python -m src.experiments.native validate RUN --instance INSTANCE`；Python 结果可用 `python main.py validate RUN`。比较时同时保留 K、分配范围、`flow_policy`、输入/源码哈希、路径范围和终止原因。模型变体之间的差异可作运营对照，不能标成同一模型的求解器优劣。

需要增补图表时按以下顺序复用：

1. **指标**：已有物理量从 `operations/` 导出表读取；新运营指标优先放入 `detail.py`，并在 `metric_dictionary` 中说明单位、分母和含义。不要在每张图里重复算服务量、机队或成本。
2. **读取和来源**：使用 `experiments/storage.py` 的 `load_seed`、`sha256`、`write_json`，或 `reporting/io.py` 的 JSON/CSV 工具。缺失、未完成、未通过核验的运行保留其状态，不补成零。
3. **图形样式**：复用 `reporting/style.py` 的 `_style()`、`_grid()`、`_save()`；`_save` 同时输出 PNG/PDF。不要另复制字体、配色、尺寸和保存逻辑。
4. **图形类型**：单次运行复用 `report.py` 的 `operating_figures`、`network_figure`；比较复用 `comparison.py`；规模实验复用 `scale_ladder_plots.py` 的准备、比较、阶段、运营和进度图函数。
5. **地图**：复用 `projection.py` / `basemap.py`，沿用坐标系与来源检查。默认本地公里坐标；需要底图时显式使用 `--cached-basemap`。
6. **落地位置**：优先给相应报告模块增函数或参数。只有输入格式或图形职责确实独立时才新增模块，并接入已有入口；不要每次在 `outputs/` 新写一份几乎相同的绘图脚本。历史冻结批次已有的报告入口继续用于该批次复查。

报告修改只需运行对应后处理和必要的读取/指标验证；无需为重画一张图重跑优化。保存新的分析目录，并记录后处理源码哈希，避免把新样式产物误认成原实验的冻结报告。

原始数据、`data/`、`MLND/` 和参考论文保持原位。旧 `mlnd_path/` 中发现的四组论文/LaTeX 文件也原位保留；该目录内旧求解器和实验脚本已删除，论文附属绘图/核验工具随论文保留。历史实验结果目录没有被用作新模型实验结论。

每次运行保存输入 JSON/NPZ、解数组、带物理节点的路径、独立核验表、报告和源码哈希。源码不再逐次复制。图形导出 PNG/PDF，地图默认使用本地公里坐标，不访问网络；显式申请缓存底图需要投影与坐标系来源，缺失时报告错误。

运行进度同时保存为最新状态 `progress.json` 和带实测时间的完整历史 `progress.jsonl`。算法比较要求物理输入和路径范围一致，K=1/K=2 分别解释。历史快照只在读取时统一模型名称，物理参数和原文件不变。

2026-10-15 的 2L/10H/50OD 南山实验入口为 `outputs/consecutive_benchmark_20261015/`，包含切片比较、逐订单聚类映射、场景配置、可重放命令和独立核验结果。

当前聚合任务流下的南山/美团 TSL 同规模 K=1/K=2 对照见 `outputs/consecutive_real_comparison_20261015/report/REPORT.md`。该实验使用两源历史 region_1 的相同面积切片、真实未缩放需求与统一处理程序，保留逐基地、柜点、OD、航段、任务组合、成本和无人机忙碌时间明细，图形同时导出 PNG/PDF。输入审计、独立结果审计、求解源码哈希与后处理源码哈希分别保存。
