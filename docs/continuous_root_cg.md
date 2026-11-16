# 外部连续松弛 CG 与全局下界

本模块在精简 Benders 主问题之外求解共同模型的连续松弛；K=1、K=2 使用同一个接口与证书流程。主问题仍只有战略二元变量和运营成本变量 θ。固定 TSL 2L/20H/50OD 实例仍为 807 个主问题变量。

## 数学对象与界的方向

记战略设计为 x、固定费用为 F(x)、运营变量为 v、运营成本为 q(v)。完整混合整数问题为

\[
z_I=\min\{F(x)+q(v):(x,v)\in\mathcal F,\ x\in X_I\}.
\]

把战略变量连续化得到 z_R≤z_I。`solve_continuous_cg` 保留 F(x)，保留共同模型的全部行、变量界及锥，只令战略变量连续且自由变化；它不是固定设计子问题，也不只优化 q(v)。`evaluate_fractional_recourse_cg` 则固定指定设计，只优化运营成本，用来生成普通割及 BDD 锚点。

连续 CG 从少量物理种子列出发，重复求解受限 SOCP、显式原锥对偶及完整标号定价。路径不直接进入锥约束，因此原锥对偶与所有线性行价格共同定义路径约化成本。已有受限原始解将遗漏路径置零后可行，故只给 z_R 的上界；它不提供整数可行上界。

每次完整定价后，对遗漏服务路径利用总任务流量不超过总需求 D，对遗漏空飞路径利用各自有限流量上界 D，向下修正对偶目标。修正沿用固定设计 CG 的完整域证明，另加有限变量界上的驻点误差修正。由此得到 L_R≤z_R，即使本轮仍存在负约化成本列，也可以保留这个有效但可能较弱的下界。未完成定价的轮次不能创建新证书；以前已认证的界仍可保留。

下界回写为一条总成本行：

\[
F(x)+\theta\ge L_R.
\]

不能写成 θ≥L_R，因为 L_R 已包含固定费用。加入行前再次保守下移并核对已验证整数上界。此行不增加变量，也不恢复 S、ρ 或运营流量变量。

## 调度与状态

- `--continuous-root-seconds` 默认 120 秒；`--continuous-root-budget-fraction` 默认 0.65。实际预算取两者与总剩余时间的最小值，包含种子、模型构造、对偶、定价和检查。
- `--continuous-root-max-columns` 默认 512，只控制本阶段每轮完整定价后加入的列数。固定设计子问题的 `--cg-max-columns` 默认仍为 64。120 秒及 65% 是基于固定 TSL 20H 短时诊断的预算安排，不是文献保证的通用最优参数；5 小时运行该阶段仍最多 120 秒。
- `--disable continuous_root` 关闭该阶段；`--plain` 也关闭它。独立连续探针的预算另外传入，不继承算法阶段比例。
- 阶段中生成的物理列可被后续原算法复用；分数设计不被当成整数 warm start 或整数解。
- 连续原始解与认证下界闭合只说明已取得共同连续模型的最优值。默认仍对主问题当前分数点进行运营成本投影分离；不能用一个标量目标地板替代整个值函数。`--disable root_projection` 才恢复先前的“连续界闭合后跳过根分离”调度。参见 [投影加强](projection_strengthening.md)。
- `continuous_root.bound_valid` 表示已有完整域下界；`pricing_complete` 表示 CG 终止认证。这两个字段并不等价。
- `primal_valid` 仅表示存在通过原矩阵检查的连续可行点。该点与连续目标不写入整数 `objective`/`upper_bound`。
- `timings.continuous_root` 是整体阶段时间，内部细分在 `continuous_root.timings`。统计总时间时不要再将内部细分重复相加。

## 数值恢复与同矩阵诊断

CPLEX 的 `Optimal` 状态不能替代原矩阵残差检查。连续点、锥对偶、完整定价和原始—对偶间隙分别验收；失败状态保留在 trace 中。

同矩阵诊断通过 `continuous_matrix_probe` 导出 C++ 原始稀疏矩阵、变量界、目标和锥，`continuous_reference.py` 逐项检查后交给 Gurobi。它与 Python 另行重建的业务模型对照是不同层面的验证。独立对偶证书可不调用求解器而重新核验，诊断所得全路径界不作为普通 CG 运行的免费输入。

数值恢复默认先尝试固定锥坐标的 LP 限制；该限制不可行或原矩阵残差检查失败时，在同一剩余预算内再尝试锥内单调盒限制。对旋转锥 `t² ≤ u v`，后者固定 `t=t0`，限制 `u≥u0, v≥v0`，并先验证非负角点满足 `t0²≤u0v0`。共享坐标、界及整个原矩阵均重新检查；不能由限制 LP 不可行推断原 SOCP 不可行。若第一阶段耗尽预算，不额外续时。

修复解只提供可行上界。限制 LP 的 `Optimal` 不证明原 SOCP 最优；继续 CG 必须重新以原锥对偶间隙认证受限 SOCP 精度，再做完整域定价。恢复不放宽原残差阈值。另有经过测试的 `extend_primal_candidate` 接口：只有新增变量零值满足其界且扩展点通过整个新矩阵检查，才接受上一点的补零扩展；该接口尚未接入生产 Benders 流程。

## 研究依据与实现边界

根先行连续松弛及分离背景见 McDaniel–Devine（1977），DOI [10.1287/mnsc.24.3.312](https://doi.org/10.1287/mnsc.24.3.312)，以及 [下界研究中的来源核查](lower_bound_research.md)。本实现选择在外部直接通过 CG 求共同连续模型，并以有效总成本行回写；不能把具体调度参数归于文献。

连续域认证依据弱对偶与当前有限流量界，具体定价见 [隐式 CG](implicit_cg.md) 和 [标号研究](label_pricing_research.md)。整数感知 BDD 另见 [BDD 诊断](bdd.md)；生产 CG 不为 BDD 自动枚举完整路径。当前没有实现单树 branch-and-Benders-cut 或完整整数 branch-and-price。

## 独立复现与锚点

```sh
make -C cpp BUILD_DIR=build-continuous-bdd continuous-probe
cpp/build-continuous-bdd/mlnd_continuous_probe examples/tsl_2l20h50od 2 120 free_K2.json --cg-max-columns 512
cpp/build-continuous-bdd/mlnd_continuous_probe examples/tsl_2l20h50od 2 120 anchor_K2.json --anchor free_K2.json --anchor-interior-weight 0.01 --cg-max-columns 512
cpp/build-continuous-bdd/mlnd_continuous_probe examples/tsl_2l20h50od 2 60 zero_K2.json --anchor zero --cg-max-columns 512
```

所有输出路径必须未使用。`--anchor` 接受设计数组或含 `design` 的结果 JSON。内点选项构造 `(1-ε)x+εx_core`，核心点采用已证明位于原战略凸包相对内部的构造，并重新检查战略约束；此操作改变锚点，不能声称已认证原最优边界点的 recourse。`anchor_construction` 保存变换说明。每个锚点仍必须独立通过完整 CG 及原锥证书，才可送入 BDD。

2026-10-21 固定 TSL 20H 实测中，K=2 的原最优边界锚点仍未完成认证；Adaptive 修复越过了原固定坐标 LP 不可行的轮次，但后续 LP 达到局部时限。此前 1% 核心点混合后的锚点完成了认证。原失败日志保留，未放宽证书阈值，也未把失败割加入比较。新的候选恢复与 LP-first 对偶策略用于两个公开连续 API，以及默认根节点分数投影评估；整数固定设计不因此改用该数值恢复流程。射线 LP 的持久复用是独立开关，任何调用进入该对偶配置时均可启用。
