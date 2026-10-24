# TSL 2L / 20H / 50OD：CSV + JSON 实例

这是冻结算例的可交换文本副本，格式为 **`consecutive-csv/1`**。它沿用旧分支的五文件布局，并增加 `model.json` 和六张系数表；**不保证旧读取器直接兼容**。导出没有重新选择点位、聚类订单、标定参数、筛选 κ 或运行优化。

布局来源为 `MLND_Path` 提交 `0fe23717de2223b0f2a21634b2d9c9db8a97d464` 的 `src/data/generator.py`、`src/data/algorithm_performance.py`；本格式的权威定义、导出、读取和校验实现在仓库的 `src/data/standard.py`。`metadata.json` 分别记录 `schema_version="consecutive-csv/1"` 和 `legacy_layout="five-file-v2"`。

## 文件与读取规则

```text
sites.csv                 候选设施：22 行
od_nodes.csv              OD 代表节点：100 行
od_demand.csv             需求：50 行
periods.csv               合并需求时段：1 行
metadata.json             数据来源、逐日数量、裁剪及标定说明
model.json                模型标量、ID 顺序、矩阵形状、单位和容量定义
coefficients/
  base_costs.csv          2 行
  locker_levels.csv       20 × 5 = 100 行
  links.csv              22 × 22 = 484 行
  od_costs.csv            50 行
  od_hub.csv              50 × 20 = 1000 行
  kappa.csv              50 × 50 = 2500 行
manifest.json             输入文件 SHA-256
verification.json         与冻结输入的精确往返校验结果
```

CSV 为带表头的 UTF-8 文本；浮点数使用可往返恢复原 double 的十进制精度，布尔值用 `0/1`。仅不存在航线的 `flight_times` 可以用 `inf`。六张系数表必须包含声明域的**全部组合**：零值、禁止航线及对角项也必须保留，不能省略为稀疏表。

按 ID 连接各表，以 **`model.json.ids` 为矩阵索引顺序**；CSV 行可以重排，不能把行号当作隐含索引。飞行节点顺序为两处基地后接二十处柜点。`L0000/L0001` 对应原报告的 `L0/L1`；`H0000…H0019` 对应 `H0…H19`，其中前十个为原候选点，后十个为同一范围内追加的候选点。OD ID 为字符串 `"0"…"49"`，对应 `O00000…O00049`、`D00000…D00049`。50 个起点节点只有 42 个不同坐标，**不得按坐标去重**。

## 五文件的含义与真实需求

| 文件 / 字段 | 定义与单位 |
|---|---|
| `sites.site_id,type` | 设施 ID；`L` 为基地，`H` 为柜点。本例没有 E 类设施。 |
| `sites.x_km,y_km`；`od_nodes.x_km,y_km` | `local_km` 局部投影坐标，单位 km；源坐标基准未确定，不应标为 WGS84 经纬度。 |
| `sites.activity_rate` | 旧布局的活动强度字段；本例未保存可直接使用的 20H 统计量，留空表示未知，不能当成 0。 |
| `od_nodes.node_id,role` | 代表节点 ID；`origin` 为取餐端，`destination` 为收餐端。 |
| `od_nodes.raw_endpoint_count` | 旧布局的节点统计字段，本例留空；真实聚类订单数见 `od_demand.order_count`，不要由此空值推断订单数。 |
| `od_demand.od_id,origin_node,destination_node` | 有序 OD 身份及其代表起终点；不是一条原始订单。 |
| `order_count,exposure_hours,lambda_per_hour` | 该 OD 簇的实际订单数、观测小时数、平均需求率；`lambda_per_hour=order_count/exposure_hours`，单位单/h。 |
| `direct_distance_km` | 簇内实际订单的直线距离均值，单位 km；不是两个代表节点之间的距离。 |
| `periods.period_id,start_hour,end_hour,dates` | 时段标识、起止小时和分号分隔的日期列表；本例每天 `[12,13)`。 |
| `periods.exposure_hours,order_count,total_rate_per_hour` | 各日时段合并后的观测小时数、实际订单总数及总平均需求率。 |

本例覆盖 **2022-10-17 至 2022-10-24，共 8 天、8 个需求观测小时**。同一地图裁剪范围内的 **17,382 单**全部进入 50 个联合 OD 簇，合计 **2172.75 单/h**，没有抽取前 50 个 OD 或缩放需求。逐日订单数为 1918、1983、2015、2044、2191、2588、2615、2028，保存在 `metadata.daily_periods`。候选点依据每天 08:00–24:00 的 **162,826 单**构造；这个数量不能代替午间需求订单数。

计数、平均距离及逐日记录来自 `outputs/consecutive_real_comparison_20261015/instances/tsl/`；几何和所有模型系数来自 `data/benchmarks/tsl_2l20h50od/`。`metadata.json` 保留源文件哈希、分区日期口径、事件小时口径、裁剪范围和聚类说明。无效时间观测不会自动从需求计数中删除。

## 模型参数与六张系数表

所有时间系数均为**小时**，需求为单/h。固定资产费用已按 **5 年 × 365 天 × 12 小时 = 21900 运营小时**摊销；不能再次除以折旧年限。

| `model.json.values` 字段 | 本例值与含义 |
|---|---|
| `n_l,n_h,n_od` | `2,20,50`：基地、柜点、OD 数量。 |
| `model_version,name` | `consecutive` 及实例名称 `tsl_L2_H20_OD50`。 |
| `max_orders` | 文件默认 2；`cases.K1/K2` 仅将其覆盖为 1 或 2。 |
| `delta` | `1/120` 小时，即 0.5 分钟的连续接单窗口；K1 的有效窗口为 0。 |
| `operating_limit` | `1/6` 小时，即完整无人机任务的平均运营时间上限 10 分钟。 |
| `fleet_cost` | `50000/21900 = 2.283105…` 元/(架·运营小时)。 |
| `flight_cost,waiting_cost` | 均为 `14.4` 元/(无人机飞行或等待小时)。 |
| `route_mode,candidate_routes` | `direct,null`：使用直接飞行路径域，没有另附限定路径列表。 |
| `flow_policy` | `aggregate`：当前聚合流量规则，允许为连续接单选择第一程路径。 |

| 系数表 / 字段 | 定义与单位 |
|---|---|
| `base_costs.base_cost` | 每处启用基地的小时固定成本，元/h。 |
| `locker_levels.level_id,slot_counts` | 容量档 1–5；总格口数分别为 8、16、24、32、40，相当于 1–5 台八格整柜。 |
| `locker_levels.slot_cost` | 该柜点整套容量档的小时固定成本：`(20000 + 40000 × 台数)/21900`，**已包含点位建设费**；首档 2.739726…、第五档 10.045662… 元/h。 |
| `locker_levels.slot_epsilon,occupancy_limit` | 允许的泊松溢出概率 0.05，以及对应的最大平均在柜件数，详见下文。 |
| `links.allowed_links,flight_times,link_cost` | 是否允许该有向航线、飞行时长（小时）、启用该航线的固定成本（元/h）；分别按起终设施 ID 索引。 |
| `od_costs.ground_cost` | 该 OD 直接骑手配送的平均单价，元/单；保留真实订单费用的簇内均值。 |
| `od_costs.deadline` | 模型中该 OD 从自身请求到最终收餐的时长上限，小时；来源为已标定的 OD 时限。 |
| `od_hub.rider_mean,first_cost` | 商家到取餐柜的前端骑手平均时长（小时）和单价（元/单）。 |
| `od_hub.last_rider_mean,last_rider_cost` | 送餐柜到顾客的末端骑手平均时长（小时）和单价（元/单），尚未按顾客选择概率加权。 |
| `od_hub.pickup_share,terminal_cost` | 自提比例，以及选择概率加权后的末端骑手费用；本例 `terminal_cost=(1-pickup_share)×last_rider_cost`，元/单。 |
| `od_hub.removal_mean,receipt_mean` | 餐品到达送餐柜后，到移出柜子、到顾客最终收餐的加权平均时长，小时；移出柜子与最终收餐不是同一事件。 |
| `kappa.first_od,second_od,kappa` | 有序连续订单兼容系数，本例为 0/1，共 851 个非零项；保留全部 2500 对，包括自身配对。κ 已按存在完整可行路径筛选，不能在导入时重新生成。 |

`occupancy_limit` 已预先算好，采用 `model.capacity_definition` 中的严格溢出定义：

```text
occupancy_limit = max μ ≥ 0，使 P(Poisson(μ) > slot_counts) ≤ slot_epsilon。
```

它限制的是**平均同时在柜的件数**，不是单/h 的吞吐量。8 格、风险 0.05 时为 **4.695227540344507 件**；40 格时为 **31.066145314494264 件**。不得将其直接当成每小时可服务订单数。Python 校验使用 `atol=rtol=1e-12`；其他实现可直接读取已给出的阈值，不必依赖 SciPy 重算。

## 读取和校验

以下命令从仓库根目录执行。`load_standard` 仅依赖本目录的 CSV/JSON，不读取源 NPZ、不重新标定，也不运行求解器：

Python 提供 `src/data/standard.py` 的 `export`、`load_standard`、`validate` 接口；C++ / CPLEX 入口 `cpp/build/mlnd_cpp --input examples/tsl_2l20h50od --k 1 --inspect` 已直接接入此格式，构建与求解命令见 `cpp/README.md`。原有 `main.py --input` 保持其原接口。metadata 中的旧路径仅用于来源追溯，独立读取不会访问这些路径。

```python
from src.data.standard import load_standard
from src.data.instances import to_planning_data

data_k1 = load_standard("examples/tsl_2l20h50od", case="K1")
data_k2 = load_standard("examples/tsl_2l20h50od", case="K2")
planning = to_planning_data(data_k2)
```

```bash
python -m src.data.standard validate examples/tsl_2l20h50od
python -m src.data.standard validate examples/tsl_2l20h50od --reference data/benchmarks/tsl_2l20h50od
```

第一条检查哈希、ID、完整表格、需求计数及容量阈值；第二条还与冻结 NPZ/JSON 精确比较。已保存的校验结果为 **41 个顶层字段、25 个数组完全一致**，K1/K2 仅改 `max_orders`，`solver_run=false`。

C++ 实现仅解析本目录的 CSV/JSON，使用显式 ID 映射和已存阈值，无需 NPZ、NumPy 或 SciPy。Python 校验器仍使用项目的 NumPy/SciPy 环境复核阈值；文件格式本身不要求这些库。`manifest.json` 保护输入文件，修改系数即不再是这个冻结实例，应另建实例并重新校验。
