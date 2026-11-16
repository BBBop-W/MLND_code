# BDD 割质量诊断：完整域内问题与同主问题比较

本模块是独立诊断工具，不替换生产 Benders，不在生产入口默认枚举完整路径，也没有宣称已实现完整 Benders Dual Decomposition 算法。给定已认证普通割，它比较连续副本、仅设施/柜档整数副本、全部战略整数副本的拉格朗日内问题；然后把实际接受的割放回同一个精简根主问题，区分截距提高与全局根界提高。2026-10-22 另增加任意乘子 oracle 和盒内 Kelley 乘子搜索入口，见下文及[研究说明](bdd_multiplier_research_20261022.md)。

实现为 [`bdd.hpp`](../cpp/include/mlnd/bdd.hpp)、[`bdd.cpp`](../cpp/src/bdd.cpp)；诊断入口为 [`bdd_probe.cpp`](../cpp/tests/bdd_probe.cpp)。CG 的锚点来源为 `evaluate_fractional_recourse_cg`，生产 CG 仍使用隐式路径域。

## 数学对象与有效性

令原问题为

\[
\min_{x\in X}\{f^\top x+Q(x)\},\qquad
Q(x)=\inf_v\{q(v):(x,v)\in\mathcal F\},
\]

其中 \(X\) 是合法二元战略域，\(\mathcal F\) 是当前共同提升模型的全部线性与二阶锥约束，\(q\) 只含运营费用。模型允许全骑手后备运营；没有场景集合或概率加权的场景追索分解。

给定完整域普通割 \(\theta\ge\alpha+\beta^\top x\)，固定其斜率，定义

\[
\phi_I(\beta)=\inf\{q(v)-\beta^\top z:(z,v)\in\mathcal F,\ z\in X\}.
\]

对任何已证明的完整域全局下界 \(b\le\phi_I(\beta)\)，均有

\[
\boxed{\theta\ge b+\beta^\top x.}
\]

证明只需取任意合法原设计 \(x\) 和其任意运营可行解 \(v\)：\(b\le q(v)-\beta^\top x\)，随后对 \(v\) 取下确界。这里不使用场景独立性，也不要求锥强对偶。强对偶/数值对偶证书仍关系到原普通割是否紧，以及如何准确衡量增强；不能由完全追索自动推定这些条件。

把全部副本改为连续，得到 \(\phi_C\)；仅设施与柜档副本保持整数，得到 \(\phi_F\)。其他战略约束全部保留，因此

\[
\phi_C(\beta)\le\phi_F(\beta)\le\phi_I(\beta).
\]

这些内问题中任意一个的完整域有效下界，都可以产生对原整数问题有效的割。部分整数化属于 BDD 内问题外松弛，**不是随机场景的 partial Benders**。

同斜率严格结构增强需要整数化后的内问题值高于连续内问题值。若原割在整数锚点 \(\bar x\) 精确紧，则由原割全局有效得 \(\phi_I\ge\alpha\)，而可取 \(z=\bar x\) 得 \(\phi_I\le Q(\bar x)-\beta^\top\bar x=\alpha\)，故 \(\phi_I=\alpha\)。因此整数紧锚点必须作为零结构增强对照。恢复原割数值下移或未紧部分，不是整数结构带来的提升。

全部整数拉格朗日割族的能力与少数限时割的效果不同。原 `bdd-probe` 只固定输入乘子；新增 `bdd-search-probe` 在有限盒内调整乘子。两者均未实现原论文的近端二次规划版本、整数 branch-and-price 或完整凸包分离。一次平行割提高也不保证根主问题的最优值提高。

## 模型和数值证书边界

`BddFullDomain` 仅接受 `catalogue_complete=true` 且 `support_complete=true` 的完整物理目录。它共享一次完整目录、一次共同模型记录构造，以及三种整数性对应的三个 Concert 模型；不同锚点复用这些对象并替换目标系数。路径枚举、共同模型、各求解器模型构造与各次目标更新均分别计时。

内问题从完整原模型构建，删除固定设计含义：副本没有 `fix` 行。战略变量原固定费用系数 \(f_j\) 被**替换为** \(-\beta_j\)，不是改为 \(f_j-\beta_j\)。运营目标与常数、原变量界、战略约束、路径列和全部锥保持不变。`facility_capacity` 保留 `DesignLayout::assignment` 之前的副本为二元，分配和航线副本改为连续；`all_design` 保留全部原战略二元性。

内问题最小化可行目标是其上界，不能用作割截距。混合整数内问题只采用完整域 CPLEX 的全局 best bound；连续对照仅在求解器最优且原矩阵残差与重算目标通过后使用其界。数值失败、`NumBest`、错误或不一致证书均不产割。零预算返回空下界，不能把默认初始化的 0 当成证书。若有限时内问题没有可行点，但具有允许终止状态及有限完整域 MIP 全局下界，该下界仍可能有效。

实际使用截距为求解器下界减去显式保守边际：

\[
b_{\rm safe}=b_{\rm CPLEX}-10^{-7}(1+|b_{\rm CPLEX}|+\|\beta\|_1).
\]

这是在浮点求解器证书之上的数值保守处理，不是独立有理数证明。输出保留求解器原界、边际和残差，拒绝与已验证内问题可行目标或适用锚点可行成本矛盾的界。锚点的 `pricing_complete` 是上游 CG 的声明；本模块检查维数、有限性、战略可行性和锚点上下值一致性，不伪称重新执行了输入割的完整定价。

只在 \(b_{\rm safe}>\alpha+10^{-6}(1+|Q_{\rm anchor}|)\) 时接受提升。对未超过门槛的情况仍报告完整域界，但 `safe_cut=null`，不会把微小浮点变化称为接受的加强割。

**不能使用受限列 MIP 的 best bound。** 缺列缩小可行域，受限内问题的下界可能高于完整内问题真值。只有完整枚举，或将来另行完成的全域整数优化证书，才支持这个截距；当前固定设计 CG 不能直接替代它。

仅全部战略整数模式经过原完整矩阵检查的可行点可以产生 `validated_original_integer_upper_bound`，其值按原目标 \(f^\top z+q(v)\) 重算。连续和部分整数化内问题的可行目标一律不输出为原整数上界。

## 同一精简根主问题比较

`compare_bdd_root_bounds` 的基准模型包含原战略约束、原固定成本、一个 \(\theta\)、当前静态解析加强，以及本次所有输入的已认证普通割。所有设计二元性在这个根 LP 中放松；不加入运营变量、锥，也不添加本次未给出的动态最大流或资源 LP 割。

先求普通割基准，再分别加入每条实际接受的 BDD 割，最后加入全部接受的割。三类比较保留相同固定成本和基准行。输出 `root_objective_change` 为相同根模型最优目标的变化；`root_gain_lower_bound` 用新模型保守下界减旧模型已验证可行目标，避免把两个近似下界的差误报为正提升。根 LP 未完成认证时，这些比较值保持空值。

这个比较隔离了本次给定割的边际作用，不是声称复现生产 Benders 全部历史割池或整次运行的 LB。它也不把拉格朗日内问题目标当作原问题的全局根界。

若生产算法已取得自由连续 CG 的全域总成本下界，还须在同一个基准和所有加割模型中加入同一行 \(f^Tz+\theta\ge L_C\)，另报这一较强基准上的边际变化。CLI 的 `--continuous-bound-file FILE` 读取 `full_continuous_cg` 输出，核对 `input_manifest`、`input_model_sha256`、`K`，并要求全域、含固定成本、`bound_valid=true`。它使用已认证下界而非连续可行目标，按生产入口再下移 `1e-8*(1+abs(bound))`。原有无 floor 比较保持不变；新增 `root_master_comparison_with_continuous_floor`。API 的可选 `certified_total_cost_floor` 参数要求调用者提供同一实例的有效总成本下界。给较弱基准加割的收益不能当作当前生产根界上的收益，单条截距提高也不能当作根界提高。

## CLI 与预算

独立构建目录避免覆盖冻结或正在运行的二进制：

```sh
make -C cpp BUILD_DIR=build-bdd-probe build-bdd-probe/mlnd_bdd_probe -j3
```

输入为 JSON 数组，或对象的 `anchors` 数组。每个元素具有以下必需字段：

| 字段 | 含义 |
|---|---|
| `design` | 按 `DesignLayout` 排列的原锚点战略向量，可以是分数。 |
| `intercept` | 普通割运营成本截距 \(\alpha\)。 |
| `coefficients` | 同顺序的斜率 \(\beta\)。 |
| `operating_objective` | 已验证锚点运营成本，不包含固定成本。 |
| `pricing_complete` | 必须为 `true`，表示上游已完成普通割证书。 |
| `name` | 可选诊断名称；省略时按数组位置命名。 |

可直接保存 `evaluate_fractional_recourse_cg` 成功返回的对象。其未认证返回含空截距，不应转换为有效锚点。实例、K、路径域和参数必须与上游锚点一致；报告保存实例 manifest、model.json SHA256 及 anchors 文件 SHA256。

```sh
cpp/build-bdd-probe/mlnd_bdd_probe examples/tsl_2l20h50od 1 anchors_K1.json bdd_K1.json \
  --seconds-per-anchor 60 --master-comparison-seconds 15 --total-seconds 195
```

每锚点的预算不超过 60 秒，由连续、部分整数和全部整数三种模式共享；顺序为该顺序，未花完的时间自动留给后续模式。根主问题比较另有默认 15 秒预算，在分配锚点求解时间前保留。总预算包含输入读取、一次完整枚举、共享模型/三个求解器构造、全部锚点及根比较；未指定时为 `60 + 锚点数*每锚点预算 + 根比较预算`。必须同时报告共享构造费用，不可仅比较每锚点优化时间。所有预算是带阶段检查的协作式截止，不是操作系统硬截止，输出保留超时量。

可以在上述命令后追加 `--continuous-bound-file outputs/root_cg_bdd_20261021/continuous_K1_initial.json`。开启后，15 秒根比较预算由无 floor 和有 floor 两组共享，先给无 floor 组一半，剩余时间用于有 floor 组；不会偷偷增加总预算。输出同时保存连续界来源文件的 SHA256、原始下界和实际 floor 值。微型回归用同一完整物理矩阵的连续最优界构造 floor，并检查两组比较保留固定成本、根界不超过独立整数最优值。

输出文件与 `.diagnostics` 目录已存在时拒绝覆盖。完成共享模型和每个锚点后保存当前报告；未完成构造时记录失败阶段，不发布不完整路径域的割。

重要输出包括每模式原始/保守 LB、已验证内问题 UB、截距提升、锚点松弛量、是否接受、求解状态、原矩阵残差、构造/优化/验证耗时。结构增益使用两种内问题有效区间计算：若连续内问题为 \([L_C,U_C]\)、整数内问题为 \([L_I,U_I]\)，则 \(\phi_I-\phi_C\in[L_I-U_C,U_I-L_C]\)，并利用非负性修正下端。不能把 `intercept_gain` 直接视作结构增益。

## 回归与独立对照

```sh
cpp/build-bdd-probe/mlnd_bdd_probe --self-test outputs/bdd_probe_validation/native.json
/opt/anaconda3/bin/python cpp/tests/bdd_reference.py \
  --native outputs/bdd_probe_validation/native.json \
  --output outputs/bdd_probe_validation/gurobi_reference.json
```

原生小测覆盖 K=1/K=2 与 aggregate/prefix 四种物理实例，分数锚点、整数紧锚点、不同分配、柜档、关闭航线和全关闭设计；检查全部三类内问题的割在其他战略处不超过精确运营成本。另检查零预算、缺列目录拒绝、未认证锚点拒绝、非整数内问题上界隔离，以及 \(\beta=-f\) 时内问题恰等于原完整目标的身份测试。根 LP 加割后的界不得超过完整整数参考，也不得因新增有效行而下降。

Python/Gurobi 脚本重新求解每个整数战略的运营参考，独立检查所有割，并核对全部整数内问题值。Python 的连续提升可能使用不同的有效航线界，因此该脚本不把两个不同连续松弛的值强行宣称相等；严格连续矩阵对照由专门的共同矩阵诊断负责。

## 2026-10-22：独立乘子搜索试验

`BddFullDomain::evaluate_multiplier` 接受任意有限系数，不依赖伪造的普通割或锚点。完整域安全下界用于主问题割；经验证的整数可行点给出乘子目标的仿射上界。两类证书分别保存，零预算不会复用旧求解界。

```sh
make -C cpp bdd-search-probe bdd-multiplier-probe
cpp/build/mlnd_bdd_search_probe examples/tsl_2l20h50od 1 anchors_K1.json result.json \
  --continuous-bound-file continuous_K1.json --method search \
  --total-seconds 600 --inner-seconds 30 --box-radius 50 --multipliers-per-root 5
```

`fixed` 对照只求给定的有限乘子池，并不在每个新根点重新生成普通 CG 割，因此不能称为完整动态 BDD1。`search` 使用有界乘子空间中的 cutting-plane LP；原文的 BDD3 使用带二次近端项的 QP。本次实现是明确限定范围的改造，不宣称已复现原文完整算法或获得无界乘子域的收敛证书。精简根主问题仍只有战略变量和一个运营成本变量。

可复现实验调度入口为 `python -m src.experiments.bdd`。它冻结代码/二进制/实例/初始证书，单线程并发不超过三组，共同初始化费用从每组 BDD 总预算扣除，另有冷启动完整 CPLEX 控制。每次求值及根解写入 `.search.jsonl`；运行中保存简明快照，结束保存完整结果。2026-10-22 的固定 TSL 20H 试验位于 `outputs/bdd_multiplier_20261022/comparison`，所有失败的初始化尝试也计费。实际性能结论见该目录的 `analysis/REPORT.md`，不能以单条截距变化代替根下界变化。

## 文献与适用范围

Rahmaniani, Ahmed, Crainic, Gendreau, Rei (2020), *The Benders Dual Decomposition Method*, Operations Research 68(3), 878–895，DOI [10.1287/opre.2019.1892](https://pubsonline.informs.org/doi/10.1287/opre.2019.1892)。本地提前在线版纸页 3–4（PDF 4–5）§2 的 Proposition 1 / Theorem 1 给出固定乘子加强；纸页 6（PDF 7）§4.2 讨论部分副本整数化；纸页 6–7（PDF 7–8）§4.3 讨论误差修正。该文主要分析和实验为 MILP/LP；本模块的锥模型有效性由上面的弱对偶式不等式直接证明，没有照搬其全局算法和速度结论。

本地来源为 `/Users/songtaowang/Zotero/storage/TRYDHFT7/Rahmaniani et al. - 2020 - The Benders Dual Decomposition Method.pdf`，SHA256 为 `2f5307538562d9e60c452b6bf05a4d4f040b8c7b3d3a43e5e39134fb50f30b96`。相关模型边界见 [下界方法研究](lower_bound_research.md)。
