# 接口与验证

唯一模型标识是 `consecutive`。旧 NPZ 可作为原始市场的读取来源，旧决策、旧路径和旧求解结果不能作为新模型解重放。

此前同一模型保存的带版本后缀输入快照在 `load_saved` 时仅统一名称，数组和物理参数保持原样，并记录名称迁移。历史文件本身不改写；新配置和新输出只使用 `consecutive`。

## 调用链

`main.py -> src.cli -> data.instances -> solvers.api -> experiments.runner -> operations.validation -> reporting.report`。

- `model/types.py`：PlanningData、InitialSegment、ServicePath、EmptyPath、PathCollection。
- `model/demand.py`：给定分配后的兼容需求重构。
- `model/resources.py`：物理路径的固定均值、费用和占用；所有准入检查。
- `model/paths.py`：直达、elementary 或给定路线目录。
- `model/compatibility.py`：优化前完整双单路径存在性筛选及各基地对支持诊断。
- `model/locker_capacity.py`：Poisson 柜格负荷阈值。
- `solvers/formulation.py`：同一个 MISOCP/fixed-design SOCP，按 flow_policy 保留逐首段或聚合任务分支等式；aggregate 按基地对和第一 OD 共享需求递推，K1 删除零窗口的续单变量，CG 增量加入路径列。
- `solvers/dual.py`：显式锥对偶、完整目录定价和全局割。
- `solvers/pareto.py`：相对内部核心点、完整对偶最优面辅助问题、数值认证和普通割回退。
- `solvers/heuristics.py`：兼容分配初始设计、固定流资产清理、单 OD 增补设计。
- `solvers/strengthening.py`：从完整路径目录计算最低任务运营成本分摊界和总柜负荷节约界；两者沿用 `courier_savings_bound` 开关，不要求分配后有正流。
- `solvers/algorithm.py`：全路径、CG、Benders 和全骑手后备。
- `solvers/api.py`：选算法、统一时间预算和停止状态。
- `data/prepare.py`：读取原订单后可配置地构造 L/H/OD，无固定数量比例。
- `experiments/runner.py`：运行和 K1/K2 对比；不覆盖非空结果目录。
- `operations/validation.py`：从物理节点独立重算，不信任保存的资源系数或汇总值。
- `reporting/report.py`：验证通过后才生成 CSV、Markdown、PNG 和 PDF。

## 输入和输出

输入以数值数组加 JSON 元数据保存。元数据保留原始来源、参数假设、场景和几何信息。模型必要字段缺失时拒绝构造。仅更新求解参数时，不重建或改变已保存的物理实例。

结果包括：

- `result.json`：objective、bound、gap、termination_reason、pricing_complete、catalogue_complete、path_scope、method、flow_policy。
- `solution.npz`：设施、柜档、OD 分配、航段、rho、idle、fleet、H、eta 和路径流。
- `path_flows.json.gz`：首段以及服务/空调位路径的物理节点与流量。
- `operations.json` 和 CSV：成本、订单、任务、基地、柜位、机队、路径和物理航段。
- `manifest.json`：输入哈希、源码哈希、环境、配置和运行状态。
- `report/`：同一保存解的独立核验报告与图形；生成报告不重新求解。

STOP 文件或信号请求中止时，保留可核验的 incumbent。目录、时间或构模规模超限明确标识；没有完整定价不能声称获得新全局割。`construction_complete` 与 `catalogue_complete` 分别记录构模和路径目录状态。

## 验证层次

单元测试覆盖 OD 守恒、非对称兼容、AA 双计数、空池、K1、Uniform 数值积分、首段共享、没有后续路径的分支、均值时限、重复式柜占用、多弧路线和预算停止。

算法测试比较完整 MISOCP、固定设计 SOCP 和两种 Benders，并在多个不同设计之间交叉检查割的下界性质，包含关闭航段与全关闭设计。
Pareto 测试覆盖 K1/K2 与 aggregate/prefix 的跨设计有效性、完整遗漏列对偶、锥约束、锚点贴合、core 更新和失败回退；增量 CG 与一次性完整构模逐列比较矩阵、目标和变量界。七项加速开关及统计口径见 `docs/algorithm.md`。
聚合精简测试覆盖完整目录定义分组、跨设计公共矩阵、κ=0 仍参与池到达率、微小正流恢复，以及零窗口变量消除。对偶停止测试区分时间预算与用户中止，确认安全回退不会丢失已验证可行解和普通割。

运营核验包括保存解形状、二元性、物理路线、缓存系数篡改、微小流量累计、所有首段与分支、订单恒等式、库存锥、机队、柜格、链路、费用及报告上下界。集成测试覆盖真实求解、保存、回读、CLI、K1/K2 对比、提前停止和出图。

原始数据和论文有独立内容指纹核对；删除旧实现不会移走或改写论文目录。

## C++ 原生算法

`cpp/src/formulation.cpp` 构造完整 MISOCP 与固定设计 SOCP 的公共矩阵；`paths.cpp` 提供首段支持、动态路径登记、贪心种子和共享资源构造器，完整枚举仅供基准/测试。`pricing.cpp` 从输入网络构造任务阶段图，按有序基地对进行精确标号定价、资源扩展和后续完成支配。`benders.cpp` 管理 RMP、显式锥对偶、可行锥射线 LP 证书恢复、加列迭代和认证割；Pareto 辅助对偶使用同一个标号 oracle 分离遗漏驻点约束。

当前 C++ 路径域为 K≤2、直接飞行。公共模型支持由初始段和输入图定义，路径 ID 动态登记，输出只保存生成列；Python 扫描定价保留为独立基准。详细契约见 `docs/implicit_cg.md`。验证入口为 `cpp/tests/pricing_probe.cpp`（标号对枚举）、`benders_reference.py`（小模型与跨设计割）、`cg_reference.py`（保存的 TSL 同设计 CG 与全列 SOCP 对照）。详见 `cpp/README.md`。

`cpp/src/strengthening.cpp` 负责输入系数推导的解析成本/资源割和运营投影松弛；`heuristics.cpp` 负责受限列 MISOCP、局部搜索与原始解验证。根节点连续设计只用于下界割，启发式的受限模型下界不会进入全局界。详见 [主问题加强](master_strengthening.md)。
