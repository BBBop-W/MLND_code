# L2 上下界增强候选 bounds-20261030

`bounds-20261030` 在 `baseline-20261029` 的配置上同时启用下面四个方法。当前已接受的 baseline 保持为 `baseline-20261029`；候选的正确性和预算内表现分别验证。H40 的资源问题和服务器测试不在本轮范围内。

| 方法 | 实现与分离规则 | 证书与预算 |
|---|---|---|
| 三态设计面总成本割 | 选一个 head，分别固定未分配、基地 1、基地 2，其他设计连续自由；完整运营域 CG 得到三个面下界，再组合总成本行 | 仅使用完整路径定价修正的认证 LB，缺证书回退全局 LB。最多占总预算 8%，单次至多 12 秒，间隔至少 60 秒，每个 head 至多尝试一次 |
| 跨基地联合队列割 | 对已认证 signed profile 选两个 head，验证全部九个状态，向其他 head 安全提升；在当前实际 master 点检验违反量 | 缓存分离累计最多占 1%；分数点分离共用已有价格搜索预算。联合割只经 master 重解后影响全局 LB |
| 设施—分配联合邻域 | 释放少数完整 head 分配组及其相连设施、所有容量档位和航线，同时保留 incumbent 的正流路径；块外分配固定 | 原模型可行性与成本核验后更新 UB；局部模型不产生全局 LB。共用已有上界搜索预算 |
| 前置初解组合 | 连续根处理前，按共享资产的增量成本选择完整正率任务分支，尝试至多三个小路径域 | 最多占总预算 3% 且不超过 18 秒，计入已有上界预算；保留全部原始任务率及骑手回退 |

四项针对 L2、K2、preprocessed、origin、aggregate 模型。依赖完整目录的方法在目录或作用域不满足时跳过；命令行不允许在不适用的需求模型或求解方法上启用这些开关。设计面割与联合队列割对整数域有效，不声称保留原来分数连续根的全部点。

```bash
make -C cpp -j2 BUILD_DIR=build-bounds-20261030
cpp/build-bounds-20261030/mlnd_cpp --preset bounds-20261030 \
  --input /absolute/path/to/instance --seconds 600 --gap 0.01 \
  --threads 1 --seed 43 --out /absolute/path/to/results
```

四个单独开关为 `--design-face-cuts`、`--queue-joint-cuts`、`--primal-coupled-neighborhood`、`--primal-seed-portfolio`。`--inspect` 可查看解析后的完整设置。旧 baseline、完整 CPLEX 和 legacy 配置默认关闭这四项；新配置后的 `--plain` 也关闭它们。

`result.json` 保留完整配置，以及 `design_face_summary`、`queue_joint_summary`、`primal_seed_portfolio_summary`、`primal_coupled_neighborhood_summary`。相应 records 保存候选、实际接受、有效性范围、费用与预算。有证书、违反当前点、加入割、最终 LB 提高是不同事件，不能合并报告。

本轮实现前快照、当前改动、测试和独立审查位于 `outputs/bounds_upgrade_20261030/`。原始数学依据见 [下界构造](l2_lower_bound_opportunities_20261030.md)。配对检查使用 H10 OD20 R04 与 H20 OD40 R03，各自同二进制、输入、seed 43、单线程、180 秒、1% gap，串行执行；这是短预算集成检查，不替代原 600 秒正式批次。
