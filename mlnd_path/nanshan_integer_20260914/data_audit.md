# 南山实例与整数战略主问题的数据审计

审计日期：2026-09-14。范围：读取六档 canonical 实例、历史最终配置与 2026-08-31 正式基准，逐文件校验并重新统计原始取餐时钟；未运行优化、未修改历史数据或模型代码。

**六档实例已经具备从 2,723 到 11,427 个真实 OD 网格市场的完整需求。最大档为 8 个基地、120 个柜、13 个独立换电点，需求 8,196.7143 单／小时。** 历史费用、速度、可用率上限和用户选择情景可用于构建新路径实验；新任务时间可接入同一批订单的支付至取餐复合历时。

## 1. 数据来源、窗口与六档规模

原始文件为[南山区运单样本](</Users/songtaowang/Documents/MLND_code/Meituan-Shenzhen/深圳市南山区运单抽样_已脱敏.csv>)，SHA-256：`0d8eb1bf6f9f23896ff73dfe88362caf36b1e994a9b36842fca9fcb5c944cbb1`。六档的几何与窗口合同见 [scale_manifest.json](/Users/songtaowang/Documents/MLND_code/data/instances/nanshan_nested_h120/scale_manifest.json:14)。

- 七个匿名日编号 1–7；候选活动窗口 08:00–24:00，共 112 小时；需求窗口 12:00–13:00，共七个小时。
- 以经纬度 `(113.94075, 22.55175)` 为共同投影中心，使用本地公里坐标。裁剪规则为原始订单两端同时位于对应正方形内。
- OD 网格 0.5 km、候选活动网格 0.25 km；每档重新聚类设施。原始订单集合嵌套，候选坐标和 OD 编号逐档重建。
- 需求使用 `order_count / 7`，保持 `demand_scale=1`，六档均保留区域与时段内全部订单。

| 档位 | 正方形边长 km | L/H/E | 起点／终点网格 | OD 市场 | 七天午间订单 | 单／小时 |
|---|---:|---:|---:|---:|---:|---:|
| H11 | 6 | 2/11/2 | 80/108 | 2,723 | 14,803 | 2,114.7143 |
| H24 | 9 | 2/24/3 | 141/197 | 6,445 | 34,348 | 4,906.8571 |
| H43 | 12 | 3/43/5 | 188/272 | 9,162 | 47,800 | 6,828.5714 |
| H68 | 15 | 5/68/7 | 212/310 | 10,571 | 54,084 | 7,726.2857 |
| H87 | 17 | 6/87/9 | 227/335 | 11,293 | 57,071 | 8,153.0000 |
| H120 | 20 | 8/120/13 | 232/347 | 11,427 | 57,377 | 8,196.7143 |

本次逐档重新读取 CSV，全部 `(origin_node,destination_node)` 唯一；每行序列化率与 `order_count/exposure_hours` 的差异不超过 `4.29e-11`。展示需求率使用精确计数除曝光小时，因此与逐行十位小数之和的微小舍入差分开记录。

六档五个 canonical 文件均匹配各自 manifest 的 SHA-256。另将 `nanshan_nested_h120` 与 `nanshan_paper_20260831` 对应目录逐字节比较：**每档的 sites、od_nodes、od_demand、periods 四个 CSV 均相同**。正式系列 metadata 更新了生成器描述及目录，旧系列仍保留 weighted-medoids 字样，正式协议使用 weighted-K-means 映射到最近活动网格的表述。当前数据内容可直接复用，元数据出处在新实验中分别记录。参见[正式数据协议](/Users/songtaowang/Documents/MLND_code/docs/paper_experiment_protocol_20260831.md:12)。

## 2. Canonical 字段与下标对应

| 文件 | 字段 | 在新路径模型中的用途 |
|---|---|---|
| `sites.csv` | `site_id,type,x_km,y_km,activity_rate` | `type=L/H/E` 分别映射基地／柜／独立换电点；坐标决定路径与接驳距离 |
| `od_nodes.csv` | `node_id,role,x_km,y_km,raw_endpoint_count` | `O...` 与 `D...` 分别保留为 \(o,d\)；记录的是聚合网格代表点 |
| `od_demand.csv` | `od_id,origin_node,destination_node,order_count,exposure_hours,lambda_per_hour,direct_distance_km` | 保留完整 OD 及真实订单率；`lambda_per_hour` 对应 \(h_{od}\) |
| `periods.csv` | `period_id,start_hour,end_hour,dates,exposure_hours,order_count,total_rate_per_hour` | 核对时段与总需求 |

加载器字段检查见 [loader.py](/Users/songtaowang/Documents/MLND_code/src/data_processor/loader.py:40)、[OD 读取](/Users/songtaowang/Documents/MLND_code/src/data_processor/loader.py:123)。距离实现为投影平面上的欧氏距离，见 [instance.py](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:106)。

`activity_rate` 来自候选生成窗口的端点活动，不能替换午间 OD 到达率。柜、基地和换电点是由订单活动生成的候选位置；数据不包含已部署设施、实际租金或换电库存记录。

H120 有 191 个起终点代表坐标重合的 OD 市场，其 `O...` 与 `D...` 仍是不同角色节点。它们保留真实需求，并可由直接骑手兜底；依据坐标重合删除会改变需求总量。

## 3. 已有参数及迁移方式

读取了 [H120 历史最终 run_config](/Users/songtaowang/Documents/MLND_code/results/nanshan_h120_final_schema5/run_config.json) 与 [H24 正式 baseline run_config](/Users/songtaowang/Documents/MLND_code/results/paper_experiments_20260831/formal/main/H24_baseline_audit_900s/run_config.json:165)。两份配置的 `model` 参数相同。

### 3.1 物理与费用

| 输入 | 历史值 | 新路径模型的处理 |
|---|---:|---|
| 无人机速度 | 20 m/s = 72 km/h | 直接用于飞行时间 |
| 可用电池航程 | 10 km | 用于两次真正补能之间的剩余电量；原值已包含运行储备 |
| 换电服务时间 | 0.05 h = 3 min | 对实际换电动作累计时间 |
| 基地开放费 | 14 CNY/h | 基地二元开放成本 |
| 柜开放费 | 1 CNY/h | 柜二元开放成本 |
| 独立换电站开放费 | 3 CNY/h | 换电站二元开放成本 |
| 飞行运营费 | 0.2 CNY/km | 计入服务／空程完整路径费用 |
| 无人机持有费 | 5 CNY/(drone·h) | 乘 idle 数量与路径流量×平均占用时间 |
| 骑手起步费 | 6.4 CNY，包含 3 路网 km | 直接骑手基准成本 |
| 骑手增量费 | 1.8 CNY/额外路网 km | 与起步费共同计算 |
| 路网绕行因子 | 1.25 | 乘欧氏距离 |
| 首／末程接驳费因子 | 0.30 | 乘相应骑手计价函数 |
| 顾客自取处理费 | 0.30 CNY/单 | 目的柜选择的自取份额对应成本 |

费用规则的原实现为

\[
c^{\rm rider}_{uv}=6.4+1.8\max\{1.25d_{uv}-3,0\},
\qquad c^{\rm feeder}_{uv}=0.30c^{\rm rider}_{uv}.
\]

源代码见 [instance.py:111](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:111)，数值出处见 [正式配置:193](/Users/songtaowang/Documents/MLND_code/results/paper_experiments_20260831/formal/main/H24_baseline_audit_900s/run_config.json:193)。

历史模型还包含航线开放费 `0.02 CNY/(km·h)` 与连接开放费 `0.05 CNY/(km·h)`。它们乘航线／连接二元变量，属于固定网络建设成本。若新模型只选设施而不选开放航线，应单独记录这两个项的处理，避免把每小时固定费直接加入每单路径费用。

### 3.2 可用率、排队与新运营参数

| 输入 | 已有记录 | 迁移结论 |
|---|---|---|
| \(\rho\) | 0.50、0.65、0.80、0.90、0.95 五档 | 连续版本可以保留上界 0.95；零服务方案允许 \(\rho=0\) |
| 柜服务率 | 30、40、50 /h | 属于旧 H 队列；新 H 为任务内操作时不沿用该队列容量 |
| 柜容量档成本 | 0.5、1、2 CNY/h | 与旧 H 服务率档配套 |
| 柜平均逗留上限 | 5 min | 属于旧队列约束 |
| 虚拟 idle 等待上限 | 5 min | 属于旧连续接单虚拟站结构 |
| 空程请求时钟 \(\kappa\) | 历史配置未给出 | 新运营输入单独设定，并记录敏感性 |
| 骑手纯行驶速度 | 历史配置未给出 | 不能由报价 ETA 直接反推；如采用距离／速度式需新增说明 |
| 载重耗电倍率、额外电池储备 | 历史配置未分解 | 新电量模型可先采用距离等价耗电；10 km 已是可用航程 |
| 每次换电额外服务费 | 历史仅含设施费与耗时 | 每次换电收费需单列情景，不能把合成实例的 0.25 自动称为历史参数 |
| 柜取放处理时间 | 原配置没有独立时长 | 与旧 H 队列平均时间区分，确定性初版可明示处理值 |

上述旧参数的含义见 [config.py:99](/Users/songtaowang/Documents/MLND_code/src/config.py:99)。原物理航段规则是直飞距离不超过 10 km，或经一个 E 后两段各不超过 10 km，见 [instance.py:351](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:351)。新任务中 H 不补能，电量跨 \(\ell\to i\to j\to m\) 连续传递；应重新检查整个任务，不能直接串接旧独立航段的可行标志。

### 3.3 用户选择

历史效用差为

\[
\Delta U_{dj}=-1+0.06(30-20)+0.35(1)-1.2d_{dj}
=-0.05-1.2d_{dj},
\qquad \eta_{dj}=\frac{1}{1+e^{-\Delta U_{dj}}}.
\]

时间单位为分钟、价格单位 CNY、步行距离单位 km；最大自取步行距离为 1.5 km。来源：[正式选择配置](/Users/songtaowang/Documents/MLND_code/results/paper_experiments_20260831/formal/main/H24_baseline_audit_900s/run_config.json:166)、[实现](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:489)。[ChoiceConfig](/Users/songtaowang/Documents/MLND_code/src/config.py:52) 将这些数值定义为未校准基准情景，可按同一情景迁移。

新路径版本可令目的柜 \(j\) 决定该比例，并折入该列末端费用。路径随机前置时间先保持在运营占用系数内；历史报价 30／20 分钟继续作为同 OD 统一服务选项属性。历史 1 CNY 折扣只进入效用差，平台是否承担补贴费用需在新目标中明确记账。

## 4. 取餐前置时间：本次按六档需求窗口重新统计

原始时钟字段为 `customer_pay_time,fetch_time,arrived_time`，精度为分钟；原始文件还含 `biz_line`。旧 canonical 生成器[读取所有业务线](/Users/songtaowang/Documents/MLND_code/src/data_processor/generator.py:65)，不区分 waimai、pinhaofan、shangou、health。

本次读取同一原始文件，用相同投影、相同六档裁剪、12:00–12:59 支付记录，统计复合时间

\[
\xi=\texttt{fetch\_time}-\texttt{customer\_pay\_time}.
\]

时间样本仅保留同日 `pay < fetch <= arrived`。不合事件顺序的记录仅从时间估计中剔除，原 OD 需求保持完整。投影与裁剪实现参照 [generator.py:433](/Users/songtaowang/Documents/MLND_code/src/data_processor/generator.py:433)。

| 档位 | 同窗口订单数 | 有效全业务时间样本 | 排除时钟记录 | 复合前置均值 min | 中位数 min | P90/P95 min |
|---|---:|---:|---:|---:|---:|---:|
| H11 | 14,803 | 14,626 | 177 | 15.0178 | 13 | 26/30 |
| H24 | 34,348 | 33,939 | 409 | 15.2137 | 14 | 26/30 |
| H43 | 47,800 | 47,258 | 542 | 15.3356 | 14 | 26/31 |
| H68 | 54,084 | 53,469 | 615 | 15.3205 | 14 | 26/31 |
| H87 | 57,071 | 56,419 | 652 | 15.3947 | 14 | 26/31 |
| H120 | 57,377 | 56,722 | 655 | 15.3936 | 14 | 26/31 |

H120 全部订单的业务线构成为：waimai 51,416、pinhaofan 2,861、shangou 2,570、health 530。只取 waimai 后有 50,905 个有效样本、均值 15.3513 分钟，与之前 [food_time_evidence 的统计](/Users/songtaowang/.codex/visualizations/2026/09/12/01a094af-d86d-7b83-94b3-b3107c6dcb7b/food_time_evidence/summary.json:191) 一致。

H120 的 232 个起点网格均有有效全业务时间样本，每格数量最少 1、最多 2,197、中位数 96；23 个网格少于 10 个样本。可以按原起点网格估计 \(\xi_o\)，对低样本网格采用统一均值或收缩。若先做确定性算法对照，各档可用上表同口径平均值；若保留一般分布，可用同批样本直接预计算 \(\mathbb E[\max(a,\xi_o+g_{oi})]\)，无需增加优化场景变量。

这个可观测复合历时同时含备货／备餐、骑手调度与到取货地点、等待和交接。原始文件没有独立实际出餐完成时间、骑手实际到店时间或商家身份编号，因此模型中将它作为“支付至实际取货”的整体随机输入最贴合字段含义。

## 5. 路径和整数变量的真实数量级

若每个 OD 可选择每个基地对，每个组可经所有有序不同柜对，忽略换电路线细分与跨阶段电量筛除，基础组合数为

\[
P_0=n_{OD}\,n_L^2\,n_H(n_H-1).
\]

| 档位 | OD 基地对二元候选 \(n_{OD}n_L^2\) | 加设施后的二元数 | 基础服务路径组合 \(P_0\) |
|---|---:|---:|---:|
| H11 | 10,892 | 10,907 | 1,198,120 |
| H24 | 25,780 | 25,809 | 14,230,560 |
| H43 | 82,458 | 82,509 | 148,919,148 |
| H68 | 264,275 | 264,355 | 1,204,036,900 |
| H87 | 406,548 | 406,650 | 3,041,792,136 |
| H120 | 731,328 | 731,469 | 10,443,363,840 |

这里按真实 OD 分开计列，同时允许 \(\ell=m\)。换电决策进一步细分路径；物理不可能或严格支配的组合可以剔除。这些是数据规模的组合计数，不是已构建优化模型的变量数。

### 5.1 现有覆盖半径与支持集

旧 `max_walk_km=1.5` 仅定义顾客自取目的柜集合 \(H_d\)，见 [instance.py:460](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:460)。起点柜的旧支持集按物理可达性和排序生成，普通到家订单使用全部物理 H-H 走廊后再做 top-k 筛选。旧模型没有“起点到柜 1.5 km”或“到家末程 1.5 km”的业务半径。

| 档位 | 每终点 1.5 km 自取柜数量范围 | 均值 | 有自取候选的 OD | 自取约束下的基础路径组合，仅 LO |
|---|---:|---:|---:|---:|
| H11 | 0–4 | 1.90 | 2,640 | 231,600 |
| H24 | 0–6 | 2.49 | 6,347 | 1,629,320 |
| H43 | 0–9 | 3.21 | 9,088 | 12,977,874 |
| H68 | 0–12 | 4.40 | 10,552 | 92,835,200 |
| H87 | 0–15 | 5.13 | 11,284 | 218,394,936 |
| H120 | 0–20 | 6.93 | 11,412 | 758,690,688 |

该表最后一列为 \(n_L^2(H-1)\sum_{od}|H_d|\)。混合服务允许到家末端，因此不能用此列替代完整 MX 路径总量。

### 5.2 物理筛除与计算筛选

本次重新检查旧 10 km 直飞／经一个 E 换电的 H-H 可达性：六档分别保留 **110、552、1,806、4,556、7,482、14,280 条**有序 H-H 弧，恰好等于各档 \(H(H-1)\)。旧单段可达性在这组地理实例中没有减少柜对。

新模型可安全采用以下处理：

1. 根据跨阶段剩余电量与实际换电点，剔除没有完整电量可行任务的 \((\ell,i,j,m)\)；采用“所有设施可开放”的放松图证明不可能，保持全局候选定义。
2. 同一 OD、基地对、柜对及设施使用签名下，删除费用和资源均不优的路径；若机队仅线性计价、没有硬上限，可按已经计入机队时间的列费用比较。
3. 路径由所有候选点上的资源标签按检验数生成，减少同时存储的列；生成点的开放状态通过正确的对偶项进入，保持全候选定价。
4. 若采用首／末程运营半径或 top-k，需要在新实例定义中明确列出规则及保留比例，并保存完整 OD 的骑手兜底需求。

历史大规模配置保留 `display_origin_choices=8`、`general_corridor_choices=10`，明确标记 `screening_is_exact=false`，见[正式配置:160](/Users/songtaowang/Documents/MLND_code/results/paper_experiments_20260831/formal/main/H24_baseline_audit_900s/run_config.json:160)及[支持集代码](/Users/songtaowang/Documents/MLND_code/src/data_processor/instance.py:474)。原论文完整索引检查使用两者均为 `all`，见[正式实验协议](/Users/songtaowang/Documents/MLND_code/docs/paper_experiment_protocol_20260831.md:35)。

## 6. 原正式基准的使用位置

H24 `baseline_audit_900s` 固定 4 线程、seed 43、目标 gap 1%，配置中原代码版本为 `5d3e2ea19429334d6ce3c9db2a53d20297de8b8c`。其[模式比较表](/Users/songtaowang/Documents/MLND_code/results/paper_experiments_20260831/formal/main/H24_baseline_audit_900s/mode_comparison.csv)记录 MX 目标 31,654.4274 CNY/h、532.6103 秒、gap 0.6297%，并标记筛选模型。这些数值属于含旧 H 队列、虚拟连续接单和离散 \(\rho\) 的历史模型。

新实验应共享 canonical 输入与有出处的参数，同时分别报告自身整数战略主问题、连续锥运营子问题、定价、总时间和上下界。模型已重构时，历史结果用于成本尺度及数据核对，算法效率对比使用新模型的同实例直接求解与分解版本。
