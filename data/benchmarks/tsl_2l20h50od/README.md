# TSL 2L / 20H / 50OD 冻结算法算例

2026-10-16 将前轮已运行的 20H 临时算例固定为算法测试输入。本目录是新的派生数据副本，未修改原始订单、论文、原实验输出或 `configs/default.json`。

## 冻结范围

- 两个基地、20 个 H 候选点、50 个 OD；总需求 **2172.75 单/小时**。
- H0–H9 是原有候选点，H10–H19 为上轮在同一裁剪范围内补充的真实订单地点；坐标见 `candidate_sites.csv`，完整地图和数据来源见 `metadata.json`。
- `input_arrays.npz`、`input_metadata.json` 与 `outputs/tsl_h20_temporary_20261015/instance/` **逐字相同**。保留元数据中的原始 temporary 名称和历史路径，以维持来源证据；本目录的正式身份由 `benchmark_manifest.json` 指定。
- K=1、K=2 共用同一物理输入，仅改变 `max_orders` 及记录该覆盖的元数据。两个已运行案例的全部 39 个其他顶层字段已与本目录核对一致。
- 当前规则：`consecutive`、`aggregate`、直接飞行；κ 的 **851** 个可行配对固定。不得重新聚类 OD、调整地图/候选点、重筛 κ 或修改成本/时间/容量后继续沿用此算例名进行算法比较。
- 每处最多五个八格整柜；整柜 40000 元、无人机 50000 元；五年、每年 365 天、每天 12 小时摊销；完整任务平均运营时间上限 10 分钟。实际全部系数以冻结数组为准。

## 固定运行条件

每个 K 单独使用 600 秒、1 线程、随机种子 43、相对 gap 0.005、列数上限 500000、最多 100 次外迭代。方法与加速开关可以变化，实例输入必须保持上述哈希。

现有 `configs/tsl_20h_benchmark.json` 保留同一物理参数，并将 `data.input_path` 指向本目录。**当前加载器把完整配置当作物理覆盖，会重新构造系数和筛 κ；严格冻结回放须使用下列 `--input` 命令，不同时传 `--config`。** 显式指定全部求解预算，避免以后默认配置漂移。

以下命令在仓库根目录执行。`inspect`、`plan` 只读取/说明，不执行优化：

```bash
python main.py inspect --input data/benchmarks/tsl_2l20h50od --max-orders 1
python main.py inspect --input data/benchmarks/tsl_2l20h50od --max-orders 2
python main.py plan --input data/benchmarks/tsl_2l20h50od --method benders_cg --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100
```

在新的、空的输出目录依次运行两个 K：

```bash
python main.py compare --input data/benchmarks/tsl_2l20h50od --method benders_cg --seconds 600 --threads 1 --seed 43 --gap 0.005 --column-limit 500000 --max-iterations 100 --out outputs/tsl_20h_algorithm/benders_cg
```

将方法替换为 `benders_cg_plain` 或 `full_path` 时，也应更换输出目录。Python 中用 `src.data.instances.load_saved("data/benchmarks/tsl_2l20h50od")` 读取；用 `dataclasses.replace(to_planning_data(data), max_orders=k)` 生成两个 K，勿调用 `prepare_orders` 或传入物理覆盖配置。

## 已有 full_path 参考结果

| 候选 H | K | 可行成本（元/h） | 成本下界（元/h） | gap | 服务量（单/h） | 终止原因 |
|---:|---:|---:|---:|---:|---:|---|
| 20 | 1 | 9975.777996 | 9925.913612 | 0.4999% | 1782.261570 | gap_limit |
| 20 | 2 | 12281.735634 | 9514.167125 | 22.5340% | 310.684789 | time_limit |
| 10 | 1 | 11189.427456 | 11134.807511 | 0.4881% | 1277.462627 | gap_limit |
| 10 | 2 | 11072.917386 | 10873.631953 | 1.7998% | 1306.036476 | time_limit |

20H 是当前算法基准；10H 仅保留为历史来源和扩点比较背景，不能混入同一实例的算法性能表。上述四次结果的 `pricing_complete=true`，飞行路径域为 `direct`，并非精确最优解。各次运行路径、输入/结果哈希、完整预算和原始摘要均在 `benchmark_manifest.json`。20H K=2 的较差可行解不能被当作最优成本。

算法实验必须共同输出可行成本、有效下界、gap、定价是否完整、路径枚举是否完整、终止原因、实际总耗时、服务量和代码/输入哈希。若定价未完成，须保留已有有效界的来源，不能将未认证的受限模型结果解释成完整问题的界。

## 完整性校验

```bash
python data/benchmarks/tsl_2l20h50od/verify.py
```

校验逐字复制文件及配置快照的 SHA-256；若历史输出仍存在，同时逐字段复核与源输入、20H K=1/K=2 输入的一致性。不会重新构建输入或调用求解器。
