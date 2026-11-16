# CPLEX 原生对偶接入

普通 Benders/CG 子问题直接读取原 SOCP 的对偶信息，不再建立或求解证书 LP 或显式锥对偶。原运营模型、分配、路径域及数值认证容差不变。资源分离 LP 与 Pareto 辅助优化仍是各自独立的强化步骤，不用于恢复普通子问题对偶。

## 接口与执行

`NativeSolver::read_dual` 返回线性行价格、变量界 reduced costs、旋转锥 dual slack 三个坐标。读取过程无 solve 调用。连续含锥模型在原始求解前设置 `Preprocessing::QCPDuals=QCPDualsForce`：默认预处理对“锥坐标由上下界固定”的模型可能不保留对偶信息，即使原始求解为 Optimal，读取也报 CPLEX 1232。固定三变量模型的默认/Force 对照和正式回归已保存。

普通行乘子按行方向校正符号；边界 reduced cost 拆为非负上下界价格；锥向量投影到旋转对偶锥。固定战略坐标的原生固定行价格及冗余边界价格不直接作为割系数，而由其余行价格和锥向量代数重构。截距不重复计入固定行 RHS。自由设计连续松弛保留全部行和界信息。

投影和舍入产生的驻点正残差按原变量有限上界修正。检查原始可行性与原始—对偶差距后，继续现有完整路径定价及遗漏列修正。完整定价仍是全域割有效性的必要步骤。原生读取或精度失败时仅重新求解原 SOCP；超过已有重试次数或预算则报告失败并保留此前有效界，不退回自建证书优化。

## 可观察契约

- 普通 CG 的 `dual_attempts[].method` 为 `native_cplex_dual`。
- `auxiliary_optimization_calls=0`；getter 耗时仍计入 dual，原 SOCP 重求耗时计入 primal。
- 普通对偶不再有 profile 0–5 的优化顺序。
- 历史 `--dual-lp-first` / `--reuse-dual-lp` 命令可解析，但不改变普通对偶来源；结果中这两项加速为 false。
- Pareto 强化失败继续保留普通割；普通对偶来源与 Pareto 的额外选割步骤分别报告。

## 验证与复现

工作开始时已有未提交的算法修改，审查边界是 `outputs/native_dual_20261028/before` 的逐文件快照，不将本轮以前的修改归入本次变更。

```bash
make -C cpp BUILD_DIR=build-native-dual -j3 all native-dual-probe continuous-probe structural-cg-cut-probe
cpp/build-native-dual/mlnd_native_dual_probe
cpp/build-native-dual/mlnd_continuous_probe
cpp/build-native-dual/mlnd_structural_cg_cut_probe /tmp/native_dual_cross_design.json
```

复用既有 M01/M07/M08 的 L2 双单实例，每例 600 秒、单线程、seed43、1%目标 gap。历史 full_refined 结果和二进制保持不变。新实验由 `outputs/native_dual_20261028/compare.py` 包装已有冻结、内存监护、求解、运营导出和审计流程。三个任务最多同时运行，单任务4 GiB、总计12 GiB。结果完整后再报告性能；原生对偶读取成功本身不证明割更强或整体更快。

## 本轮验证结果

- 原生 getter 的解析 LP、增量列、旋转锥正负坐标、零坐标、锥顶、固定锥坐标边界回归通过。
- 既有结构 CG 测试通过 2160 次跨设计割检查、32 次关闭基地检查；最大跨设计违反为 0，固定设计目标最大差约 1.22e-7。
- 连续根松弛、分数设计及隐式完整路径验证通过。统一任务的单单/双单端到端测试通过，并检查每个普通对偶记录均为 native getter、额外优化次数为 0。
- 70 项相关 Python 输出、审计与实验协议回归通过。最终小测试清单见 `outputs/native_dual_20261028/VALIDATION.json`。
- 双轴审查无阻断项；旧 LP-first 初始化注释已移除。本轮专用差异为 `outputs/native_dual_20261028/CHANGE.patch`，不含此前积累的工作区修改。

生产对照结果单独保存在 `outputs/native_dual_20261028/comparison/analysis/REPORT.md`。M01 在已运行的一轮中出现原生对偶有效但修正后不够紧的早停；直接读取并不消除大乘子的数值认证问题，不能把这个终态归为正常达到 gap 或正常限时。
