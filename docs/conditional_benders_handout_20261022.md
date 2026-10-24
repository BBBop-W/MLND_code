# 固定分配需求计算与条件 Benders 实现手册

本手册交给负责代码的 agent，用于实现并测试一种替代当前 lifted Benders 的算法。目标是在不改变当前模型的前提下，固定 OD 分配后在优化模型外计算需求重构系数，删除运营子问题中的需求线性化辅助变量，再用按分配模式生效的条件对偶割连接整数主问题。

交付应是一个可以独立启停的实验模式，以及与当前代码逐项对照的正确性和性能结果。本文提供实现规格；新模式尚未实现，本文中的拟新增函数、参数和输出字段均不是现有 API。原算法继续作为基准。

## 1 本次实验的边界与代码基准

以当前工作区代码为唯一模型基准。论文、其他窗口中的草稿、讨论中的新概率公式，均不进入这次实验。

- 仓库：`/Users/songtaowang/Documents/MLND_code`。
- 分支：`MLND_Continuation`。
- HEAD：`0fe23717de2223b0f2a21634b2d9c9db8a97d464`。
- 工作区包含尚未提交的改动；HEAD 本身不能完整标识模型。相关 30 个文件的 SHA256 已记录在 [代码基准清单](/Users/songtaowang/Documents/MLND_code/docs/conditional_benders_handout_20261022_baseline.json)。实施前核对该清单；如果代码继续变化，记录实际实施基线，不覆盖其他 agent 的改动。
- 当前模型为 `consecutive`，支持 `max_orders=1,2`；默认 `flow_policy="aggregate"`，同时保留 `flow_policy="prefix"` 这一现有配置值。
- 当前原生 C++ 隐式定价支持 `route_mode="direct"`、`candidate_routes=null`。本实验不扩展物理路径域。
- 时间、流量与费用沿用当前输入单位：小时、每小时到达率、对应小时成本。

下列内容保持原样：

| 内容 | 本轮处理 |
|---|---|
| 需求概率 | 保持当前 `q * kappa * share` |
| 兼容矩阵 | 使用输入中已有的矩阵，不按当前选址或生成列池重新计算 |
| 全部兼容、兼容率 `A_br`、运营策略开关 | 不引入 |
| 任务最多订单数、同 OD 双单 | 保持现有 K 和 AA 语义 |
| 两种流量策略 | 分别保持 `aggregate` 和 `prefix`，不互相替换 |
| 路径时间、骑手、交付时限、格口、能耗口径 | 完全沿用当前物理系数及筛选函数 |
| 基地库存锥、机队、空载调位 | 保持现有约束与成本 |
| 选址主问题 | 保留现有整数变量、战略约束和运营成本代理变量 |
| 连续根节点、Pareto、邻域搜索等加速 | 首轮采用新旧一致的 plain 设置，后续再单独适配 |
| 原始数据、论文、默认配置 | 不因这项算法实验修改 |

本轮只研究算法：固定同一个整数设计，新旧运营模型的最优成本必须一致。新算法不需要重新解释运营机制，也不需要修改文章。

## 2 文献依据与这份规格的关系

**Pantuso（2025），Solution of stochastic facility location problems with combinatorially many decision-dependent distributions，Journal of Combinatorial Optimization。** §3 的 Algorithm 1 先解选址主问题，识别其诱导的分布，再只求对应运营子问题。Proposition 1、式 (4) 给出按分布模式生效的对偶割；后续命题讨论有效性和收敛。可借鉴的是按需评价和条件割，不引入它的随机需求场景模型。[开放全文](https://link.springer.com/article/10.1007/s10878-025-01371-7)

**Pantuso 与 Hewitt（2025），The L-shaped method for stochastic programs with decision-dependent uncertainty，Mathematical Programming。** §3.1.2、式 (8) 是分布特定对偶割；§3.2 另讨论可计算值函数的整数最优性割。两类工具不要求同时使用。本文采用前一类原则，将需求模式映射为当前二元 OD 分配，并处理 SOCP 证书。[开放全文](https://link.springer.com/article/10.1007/s10107-025-02246-9)

后文的具体消元、代码接口和测试是针对当前仓库的推导与实现规格。第一版只使用条件对偶割，不把整数点值割列为必要组件。LP 文献依靠有限对偶极点的证明不直接套用于 SOCP；本模型的精确算术有限终止依据见第 8 节。

## 3 总体分解

设完整战略向量为 `x=(alpha,y)`。`alpha` 是全部 OD 到有序基地对的分配；`y` 在本手册中统指其余战略变量，并不只是代码里名为 `airline` 的数组。连续运营变量统记为 `z`。

目标仍为：

\[
\min_{(\alpha,y)\in\mathcal X}\{F(\alpha,y)+Q(\alpha,y)\}.
\]

`F` 是固定设施和航段成本，`Q` 是运营成本。主问题保持：

\[
\min F(\alpha,y)+\zeta,
\qquad (\alpha,y)\in\mathcal X,
\qquad 0\le\zeta\le C_0,
\]

其中 `C0` 是当前全骑手运营成本。逐轮加入条件割。所有真实连续运营决策仍留在 SOCP。

```text
整数主问题产生 alpha 和其余设计 y
                  ↓
按 alpha 计算 lambda、q、share、P、单单系数
                  ↓
建立或更新精简 SOCP，固定本次完整设计
                  ↓
原始求解 → 锥对偶 → 全候选路径域定价 → 加列
                  ↓ 完成证书认证
将对偶仿射界改写为按 alpha 模式启用的条件割
                  ↓
更新上界、主问题下界和停止条件
```

### 3.1 第一版选择保留原来的任务行

最小改动方案是：**保留 `head`、`group_head`、`v` 及现有全部任务分支和订单守恒行，只删除 `eta/u/Z/Y` 变量及对应递推、乘积和连接行。**

外部计算的参数是 `P` 和单单系数等。重构的 `theta,nu` 可另行计算作诊断，但生产求解不必为消元再求一次稠密线性系统。现有线性订单守恒会自动给出相同的任务起始量。

这一选择同时适用于当前两种流量策略。不要在第一版把所有约束改写成 `nu*rho`：如果不慎把汇总任务率强加给每条首程，或者只保留总任务率，会改变逐首程比例或首程执行关系。

### 3.2 新旧算法的区别

| 项目 | 现有 lifted Benders | 本次条件 Benders |
|---|---|---|
| 主问题战略变量 | 现有设计变量 | 相同 |
| 需求重构 | 用辅助变量表达分配依赖 | 固定整数分配后计算数值系数 |
| 子问题矩阵 | 跨全部设计统一 | 同一 alpha 统一，不同 alpha 可以改变 |
| 对偶仿射界 | 对全部合法设计有效 | 首先对同一 alpha 的全部 y 有效 |
| 返回主问题 | 普通全局割 | 加上 alpha 模式条件后的全局有效割 |
| 分数 alpha 调用 | 原全局松弛允许 | 第一版拒绝 |

## 4 外部需求系数

权威参照为 [demand.py](/Users/songtaowang/Documents/MLND_code/src/model/demand.py:28) 的 `demand_rates`，并与当前 formulation 中的递推对应。

对池 `b=(l,m)`，固定整数分配后：

\[
d_{rb}=h_r\alpha_{rb},\quad
\lambda_b=\sum_r d_{rb},\quad
q_b=-\operatorname{expm1}(-\delta_{\rm eff}\lambda_b).
\]

当 `lambda_b>0` 时：

\[
\pi_{sb}=\frac{d_{sb}}{\lambda_b},\qquad
P_{brs}=q_b\kappa_{rs}\pi_{sb}.
\]

采用当前 `demand_rates` 中稳定的单单系数：

\[
s_{br}=e^{-\delta_{\rm eff}\lambda_b}
+q_b\sum_s(1-\kappa_{rs})\pi_{sb}.
\]

数学上 `s_br=1-sum_s P_brs`。数值实现使用上面的稳定公式，减少 `q≈1` 时的消减误差。

边界约定：

- 空池 `lambda_b=0`：`q=0, share=0, P=0, s=1`。
- K1 使用当前 `effective_delta=0`；不建立原模型没有的双单行。
- K2 且原窗口为零，也沿用当前 `effective_delta=0` 行为。
- `kappa=0` 的 OD 仍然进入池的 `lambda_b`，因为当前模型使用全部已分配订单率。不得改成兼容订单总率。
- `kappa` 可以是非对称小数，不能只按 0/1 分支实现。
- 不通过阈值把小的正概率改成零，也不在输出中抹掉小的正流量。

参数计算接口可以只保存每池的 `lambda/q/share/s`，按需返回 `P_brs=q*kappa[r,s]*share[s]`，不必永久分配一个额外的稠密 `B*R*R` 数组。第一版不做支持集裁剪，先确保与旧行结构一致。

如计算诊断用任务率，应使用线性方程求解而不是显式求逆：

\[
(I+P_b^\top)\theta_b=d_b,\qquad
\nu_b^r=s_{br}\theta_{rb},\qquad
\nu_b^{rs}=\theta_{rb}P_{brs}.
\]

它们应满足 `single + pair.row_sum + pair.column_sum = assigned demand`。AA 在行和、列和中分别贡献一次。这个恒等式用于审计，不替代现有物理路径和任务分支约束。

## 5 精简 SOCP 的逐行替换

令 `g` 为当前代码中的一个组，`r(g)` 为其首单类型，`b(g)` 为其基地对，`H_g` 为该组的起始流量。

- `aggregate`：一个组由相同 `(l,m,r)` 的首程组成，`H_g=group_head[g]`；保留 `group_head=sum(head_j)`，以及每个首程的 `executed_head` 行。
- `prefix`：每条首程单独成组，`H_g=head[j]`；保持当前没有 aggregate `executed_head` 行的结构。

记 `P_g^single`、`P_g^double(s)` 为这一组的单单和有序双单路径集合。只替换下列三类式子。

**单单分支：**

\[
s_{b(g),r(g)}H_g-
\sum_{p\in\mathcal P_g^{\rm single}}\psi_p=0.
\tag{C1}
\]

**双单分支：**

\[
P_{b(g),r(g),s}H_g-
\sum_{p\in\mathcal P_g^{\rm double}(s)}\psi_p=0.
\tag{C2}
\]

**池内 OD 守恒：**

\[
\sum_{g:b(g)=b,r(g)=r}H_g
+\sum_{g:b(g)=b}P_{b,r(g),r}H_g
-h_r v_{rb}=0.
\tag{C3}
\]

保留原三行 `v<=rho, v<=alpha, v>=rho+alpha-1`，以及 `v` 的现有上下界。固定二元分配时仍有 `v_rb=alpha_rb*rho_l`。

路径列在分支行中的系数仍为 `-1`；每个 OD 的 `served_counts`、柜格占用、机队忙碌、基地平衡、航段使用次数均不变。因此，保留行映射后，当前 `path_column` 可以继续作为物理列的权威接口。

### 5.1 必须保留的零支持行

即使某个正概率双单分支没有可行路径，也保留 (C2)。它会令对应 `H_g=0`，并可能通过共享基地可用率影响其他任务。不得删行、删需求、置零兼容性或改成骑手续单分支。

某个 OD 没有合法首程时，仍保留它在当前模型中的全部 `orders_{r,l,m}` 行。第一版不按当前开放设施或当前列池缩减任务支持结构。

当前模型允许所有订单由直接骑手完成，故没有无人机路径不等于运营不可行。合法整数设计下可取 `rho/head/flows/idle/fleet=0`、`courier=h`，锥辅助变量按旧模型取值。这一完整补救性质应通过现有基线验证继续确认。新模式若在这种设计下报告 infeasible，应先排查消元、索引或数值问题，不能立即加 no-good 割排除方案。

### 5.2 本轮具体删除什么

删除每组 `eta`、`u`、`Z`、`Y` 的创建及下列行：`eta_upper`、`chain_first`、`u_eta/u_assignment/u_lower`、`Y_chain/Y_assignment/Y_lower`、`chain_next`、`eta_link`。

不要删除组起始流量连接、分支、OD 守恒、基地库存锥或任何物理资源行。

若 `delta_eff>0`，当前每组有 `3R+2` 个待消去变量、`7R+3` 条待消去线性约束，因此：

\[
\Delta n_{\rm var}=G(3R+2),\qquad
\Delta n_{\rm row}=G(7R+3).
\]

锥数不变。双单分支行和路径数仍可能为二次或更大规模，不声称整个子问题变为线性规模。K1 原本没有这批变量，不预期有这部分规模收益。

## 6 为什么这是精确消元

以下证明针对当前模型和固定二元分配，不针对尚未采用的新需求公式。

旧模型的二元乘积约束给出：

\[
u_{gs}=\alpha_{sb}\eta_g,\qquad
Y_{gs}=\alpha_{sb}Z_{gs}.
\]

以代码中的 OD 顺序定义累计量，旧递推得到：

\[
Z_{g,j}=H_g\exp\!\left(-\delta_{\rm eff}
\sum_{s<j}h_s\alpha_{sb}\right).
\]

最后一个节点和 `eta_link` 给出：

\[
\lambda_b\eta_g=q_bH_g.
\]

若 `lambda_b>0`，则 `eta_g=(q_b/lambda_b)H_g`。代入原来的单单、双单、OD 守恒式，恰好得到 (C1)–(C3)。所有保留变量、物理流和成本不变。

反向从精简模型解恢复旧辅助量：

\[
\eta_g=\frac{q_b}{\lambda_b}H_g,\quad
u_{gs}=\alpha_{sb}\eta_g,\quad
Z_{g,j}=H_g e^{-\delta_{\rm eff}\sum_{s<j}h_s\alpha_{sb}},\quad
Y_{gs}=\alpha_{sb}Z_{gs}.
\tag{R}
\]

旧变量界也没有丢失。由 (C3) 的非负项：

\[
H_g\le h_{r(g)}\alpha_{r(g),b}\rho_l.
\]

首单未分配时 `H_g=0`；已分配且有正需求时 `lambda_b>=h_r`，故：

\[
\eta_g\le\frac{q_bh_r}{\lambda_b}\le1,
\qquad
\eta_g\le\delta_{\rm eff}H_g\le\delta_{\rm eff}h_r.
\]

因此原来的 `eta<=min(1,delta_eff*h_r)`、`eta<=delta_eff*H_g` 及 `u/Z/Y` 上界均得到满足。零原始需求对应 `H_g=0`。空池时 (C3) 的非负性迫使全部 `H_g=0`，旧 `eta_upper` 又迫使 `eta=0`，所有辅助量恢复为零。

所以新旧模型在保留变量上的可行域投影和目标完全相同。该证明覆盖 aggregate 和 prefix，不需要额外引入需求近似。

### 6.1 对公共结果格式的恢复

公共结果中的 `_solution.eta` 是按首程导出的，不是按组。两种流量策略都应导出：

\[
\eta_j=\begin{cases}
(q_b/\lambda_b)\,head_j,&\lambda_b>0,\\
0,&\lambda_b=0.
\end{cases}
\]

不要将一个 aggregate 组的 eta 复制到每个首程。现有 C++ 输出函数会跳过 `eta` 索引为负的组；只删除变量而不修改输出，会错误地产生全零 eta。

`u/Z/Y` 不需要重新放回优化模型或公共 JSON，只在旧矩阵审计时按 (R) 恢复。保留现有独立运营验证器，不能通过关闭旧检查让新模式通过。

## 7 条件对偶割的构造与插入

### 7.1 固定分配下的仿射界

第一版继续保留现有设计副本变量、设计约束及全部 `fix_i` 行，减少对索引和对偶接口的改动。需求系数固定为 `P(alpha^k)`。

同一 alpha 下，不同 y 的矩阵、目标系数和变量界必须保持相同，只改变设计固定行的右端。对偶经过现有有限界修正及完整定价后，得到：

\[
\widehat a_k+\sum_i\widehat\beta_{ki}x_i.
\]

这个数值系数模型的原始仿射界不能直接当成跨全部 alpha 的原问题割。先将 alpha 对应项在锚点吸收到截距：

\[
a_k=\widehat a_k+
\sum_{i\in I_\alpha}\widehat\beta_{ki}\alpha_i^k,
\qquad
g_k(y)=a_k+\sum_{i\notin I_\alpha}\widehat\beta_{ki}y_i.
\tag{D1}
\]

在当前原生布局中：`I_alpha = [layout.assignment, layout.airline)`；flatten 顺序为 `(r*L+l)*L+m`。吸收后把这部分原始系数清零，再添加下面的模式条件。

### 7.2 条件项和安全大 M

令 `A1={i:alpha_i^k=1}`、`A0={i:alpha_i^k=0}`，完整 alpha 位串参与匹配，包括未分配项：

\[
\Delta_\alpha=\sum_{i\in A_1}(1-\alpha_i)+\sum_{i\in A_0}\alpha_i.
\]

当前成本非负，可取全局运营下界 `L0=0`。所有 y 都在 `[0,1]`，故可以直接用盒上界：

\[
U_k^{\rm aff}=a_k+\sum_{i\notin I_\alpha}
\max(\widehat\beta_{ki},0),
\qquad
M_k=\max\{0,U_k^{\rm aff}-L_0\}+\epsilon_k^{\rm gate}.
\tag{D2}
\]

`epsilon_gate` 是向上取整和浮点累加的保守余量，记录其值。这里不需要再解一个 MIP。不要把当前运营成本 `Q_k` 直接当成 `M_k`，它不是仿射函数在其他设施方案上的上界。

最终割为：

\[
\boxed{\zeta\ge g_k(y)-M_k\Delta_\alpha.}
\tag{D3}
\]

同一 alpha 时它是对偶下界；不同整数 alpha 时右端不超过 `L0`。整个割对所有整数战略方案有效，因此可以正常加入 MILP。

### 7.3 转成当前 Cut 数组的明确公式

假设 `keep_cut` 采用 `zeta >= intercept + coefficients dot design`，则生成新对象：

```text
conditional.intercept = a_k - M_k * |A1|
conditional.coeff[i] = beta_hat[i]       if i not in I_alpha
conditional.coeff[i] = +M_k              if i in A1
conditional.coeff[i] = -M_k              if i in A0
```

必须以单元测试检查三个事实：锚点处恰好回到经修正的 `g_k(y^k)`；任意改变一个 alpha 位后，右端不超过 `L0`；同一 alpha 下变动 y 时仍得到 `g_k(y)`。

建议新增元数据：`cut_kind="conditional_dual"`、`cut_scope="fixed_assignment_all_facility_designs"`、`assignment_fingerprint`、`gate_M`、`affine_upper_bound`、`gate_rounding_margin`、`anchor_operating_primal`、`anchor_certified_lower`。这些字段为拟新增接口。

不要把 compact 原始 Cut 先交给现有无条件 `keep_cut`，之后再补标签。应在完成条件化以后才插入主问题；缺少模式元数据时拒绝插入。

### 7.4 简单数值示例

设 `alpha^k=(1,0)`，`g(y)=100-20y1-30y2`，`L0=0`。忽略数值余量时取 `M=100`。最终割为：

\[
\zeta\ge100\alpha_1-100\alpha_2-20y_1-30y_2.
\]

当 alpha 为 `(1,0)`，不同 y 分别得到 100、80、70、50 的下界；其他 alpha 下右端均不超过零。这只是割代数的测试例，不是项目的实验数据。

## 8 证书与精确性

### 8.1 理想算术中的结论

若每次主问题全局求解，每次子问题精确求值，条件对偶割全局有效且在当前完整整数设计处取等，全部战略设计有限，则只用 (D3) 就足够有限终止。同一完整设计被评价以后不能再以低估的运营成本出现；再次成为主问题最优候选时即可结合上界认证最优。未终止的迭代必须评价新的完整设计。

实际主问题若因时间或 gap 限制只返回一个整数 incumbent，重复设计本身不能证明最优。此时仍须以主问题的认证下界计算全局 gap；只有下界与当前上界达到停止标准时才能报告相应精度的最优性。

第一版不需要再加整数点值割。仅有有限种 alpha 而 y 连续，不足以按这个论证保证 SOCP Benders 有限步终止；当前代码的 y 是二元设计，符合这里的有限设计前提。

### 8.2 继承当前数值证书

实际浮点计算仍以当前证书流程为准：行乘子符号、锥可行性、驻点残差、有限变量界修正、遗漏列约化成本修正、原始可行性和锚点间隙全部核验。

先对仿射界作安全下移，再计算 (D1)–(D3)。用于残差修正的变量界必须对同一 alpha 下的所有合法 y 有效，不能用当前开放网络或当前最优流产生的更小界。

经修正的割在锚点可能略低于最优成本。保留当前容差和 `certificate_tolerance_limit` 等真实停止原因，不能将理想有限终止证明写成浮点运行必定零 gap。

### 8.3 列生成的覆盖范围

第一版直接复用全候选物理路径域，不缩减定价域。特别是：

- 当前关闭的设施或航段，可能在同一 alpha 的另一 y 中打开，相关路径必须进入对偶定价覆盖。
- 保留所有相关空载调位路径。
- 路径物理系数、航段流上界沿用当前输入域推导，不能从当前生成列的极值估计。
- 定价的批量加列上限不等于允许忽略其他路径。
- 超时、用户停止或未完成定价时，不能发出新的条件割。

完整定价后，继续调用当前遗漏列误差修正。受限 SOCP 的可行解能更新全局 UB，但其最优值通常只是完整运营问题的上界，不能未经定价拿来作为点值割或完整成本下界。

参数模式只依赖 alpha，以当前固定 kappa、窗口、物理输入为前提。未来若这些参数依赖 y，需要扩大模式，不能沿用本次模式定义。本实验不做这样的模型变更。

## 9 代码接入位置与接口

### 9.1 原生 C++ 主线

| 位置 | 职责与实施要求 |
|---|---|
| [formulation.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:30) `DesignLayout`、`design_model` | 保留主问题设计变量顺序与战略约束 |
| [formulation.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:78) `build_formulation` | 保留旧构造器，新增 opt-in compact 分支或独立构造器；按第 5 节替换 |
| [formulation.hpp](/Users/songtaowang/Documents/MLND_code/cpp/include/mlnd/formulation.hpp:11) `Formulation` | 保持组、首程、分支和资源行映射；增加模式标识及外部系数存储 |
| [formulation.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/formulation.cpp:240) `path_column`、`add_columns` | 原物理列系数应可原样复用，按测试确认 |
| [benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:132) `solve_dual` | 复用显式锥对偶和有限界修正；标记 compact 对偶只具固定分配含义 |
| [benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:375) `correct_missing_columns`、`certify` | 继续要求完整定价和数值证书 |
| [benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:407) 内层 SOCP 和 CG | 根据 opt-in 模式选构造器，保持同样的定价、恢复和停止处理 |
| [benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1332) `solve_benders` | 主问题仍是设计加 `operating_cost`；在入割前条件化 |
| [benders.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/benders.cpp:1487) `keep_cut` | 不允许将未条件化的 compact 仿射界按 ordinary global cut 插入 |
| [output.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/output.cpp:195) `solution_json` | 从实际 head 恢复每首程 eta，兼容现有输出 |
| [heuristics.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/heuristics.cpp:25) `lifted_point` | 旧矩阵验证使用；不要直接假定它适用于删变量后的新矩阵 |
| [main.cpp](/Users/songtaowang/Documents/MLND_code/cpp/src/main.cpp:169) `--plain` | 首轮新旧算法一致关闭默认加速 |

上述行号是本手册基线定位，实施时以函数名和基准清单核对。当前主程序方法为 `full_path`、`benders_full`、`benders_cg`，尚无 conditional 选项。

建议新增独立配置 `--demand-formulation lifted|conditional`，默认 `lifted`；这是建议名称，尚不可直接调用。也可用单独实验入口，但必须把选择写入结果。`conditional` 只接受整数设计；在 free-design 完整 MISOCP 和分数 root 调用上明确拒绝，不静默退回另一模型。

### 9.2 参考 Python 与验证器

- [src/model/demand.py](/Users/songtaowang/Documents/MLND_code/src/model/demand.py)：数值重构参考。
- [src/solvers/formulation.py](/Users/songtaowang/Documents/MLND_code/src/solvers/formulation.py)：旧有限路径模型及行对照。
- [src/solvers/dual.py](/Users/songtaowang/Documents/MLND_code/src/solvers/dual.py)：有限目录证书参考。
- [src/operations/validation.py](/Users/songtaowang/Documents/MLND_code/src/operations/validation.py:426)：独立检查 eta、任务比例、订单守恒及物理量。
- [src/experiments/native.py](/Users/songtaowang/Documents/MLND_code/src/experiments/native.py)：原生结果读取与复验。

生产实现优先放在当前原生链路；Python 可以只作数学和小目录验证，避免同时重写两个生产算法。原验证器的检查与容差保持不变。

### 9.3 模式缓存

需求参数缓存键至少包含实例及模型指纹、完整 alpha 位串、K、有效窗口、kappa、flow_policy 和物理路径域标识。同一实例一次运行中，可把不变项提升为公共键前缀。

可以跨模式保存物理路径记录，但行索引、模型矩阵、对偶价格、证书和已修正仿射界不能脱离其模式复用。第一版两种算法使用相同的列复用设置；不要同时改变缓存策略来制造不公平的速度比较。

## 10 规范伪代码

下面是实施规格，不是可直接调用的现有接口。

```text
initialize the unchanged integer master
initialize a validated all-courier incumbent and its total UB
initialize LB from a certified master bound

while budget remains:
    solve master; record its certified bound, not its incumbent as LB
    obtain an integer design xk = (alphak, yk)
    verify design integrality with the existing tolerance

    params = parameter_cache.get_or_compute(alphak, instance_fingerprint)
    # params uses current q*kappa*share, not compatible-arrival A_br

    build compact SOCP at params with existing head/group/v and fix rows
    repeat:
        solve current restricted SOCP
        validate any primal candidate; update UB if genuinely feasible
        solve/repair the explicit conic dual
        price the full physical path domain using compact row maps
        if interrupted: keep valid incumbent; issue no new cut; exit safely
        if negative columns: add columns and continue
        apply full-domain missing-column correction and gap checks
        certify current operational lower bound; finish inner loop

    absorb alpha fixing coefficients into affine intercept
    compute safe affine box upper bound and gate M
    form conditional cut and verify its anchor and metadata
    insert only the conditionalized cut into master

    compare global UB and certified LB
    if tolerance met: finish with certified gap
    # a repeated incumbent alone does not certify optimality
    if repeated design cannot close corrected certificate gap:
        preserve the actual numerical termination reason
```

实际实现可以在主问题搜索过程中分离整数候选，但第一版采用现有外层迭代入口即可；不同时更换 callback 架构。

## 11 分阶段实施与验收

### 阶段 A 固定设计消元

实现外部系数计算器、compact 构造器、eta 输出恢复。只测试同一完整整数设计下的旧全列 SOCP 与新全列 SOCP。

验收：

1. 运营最优值在原有容差内一致。
2. 新解按 (R) 恢复后通过旧矩阵检查，成本不变。
3. 旧解去掉辅助量后通过新矩阵检查。
4. 两种模型的物理结果分别通过同一独立运营验证器。
5. 变量和行数减少量符合第 5.2 节；锥数不变。

不要求两个求解器返回完全相同的最优流向量或机队分解，多重最优解可以不同。用双向映射检查可行性，目标比较用各自证书区间和现有数值容差。

### 阶段 B 条件割

在完整有限路径小实例上生成 compact 对偶界，再按第 7 节转换为条件割。先不启用隐式定价。

对一个 anchor，穷举合法设计：同一 alpha 下检查未 gate 的 `g(y)<=Q_old(alpha,y)`；不同 alpha 下检查 gated RHS 不超过 `L0`。保存最大违反量及相应设计。检查 anchor 成本差不超过证书声明的修正和容差。

同一 alpha、不同 y 的模型矩阵、变量界和目标应相同，只有固定行 RHS 改变。不同 alpha 的矩阵允许不同；旧算法的跨所有设计矩阵相同测试继续保留，但不套给新模式。

### 阶段 C 内层列生成

将 compact 构造器接到现有内层 CG。比较 compact 全列与 compact CG，核对路径语义、输出、价格覆盖和完整证书。

必须设置一个 anchor 关闭某条有价值航段、另一同 alpha 设计打开该航段的测试，保证定价没有按当前 y 删除潜在路径。

### 阶段 D 外层算法

新旧都采用 plain 设置。小实例比较完整 MISOCP、旧 Benders、新条件 Benders，三者应达到相同最优值和指定 gap。保留原算法作为可复现基准，不替换默认行为。

### 阶段 E 性能比较

通过 A–D 后才运行更大的冻结实例。先用同一组固定设计隔离每次子问题的开销，再运行完整外层比较。最后才研究新模式的列复用、参数缓存和其他加速。

首轮关闭 continuous-root、root-cut/projection/inout、ordinary Pareto、restricted-MIP、local branching 等依赖全局 lifted 矩阵的流程。可以在未来单独保留旧根界作为全局加强，但那是另一个实验组，须计入完整时间和内存，不混入“全部消除 demand lifts”的首轮结果。

## 12 必需的测试资产与反例

以下测试是针对数学等价和全局割的必要检查，不是仅复刻实现。

| 测试 | 直接可用的当前资产 | 要捕获的错误 |
|---|---|---|
| K1、K2、零窗口、空池、非对称小数 kappa、AA | `tests/unit/test_model.py`、`test_pooling.py` | 概率、单位、AA 重复计数或边界错误 |
| 不兼容 OD 仍进入 lambda | `test_pooling.py::test_zero_kappa_od_still_enters_the_assignment_arrival_chain` | 偷换成兼容到达率 |
| 极小正流量 | `test_pooling.py::test_positive_tiny_heads_use_actual_head_sum_and_are_not_erased` | 输出截断和 eta 恢复错误 |
| aggregate 与逐首程差异 | `test_task_relaxation.py::route_tradeoff_data` | 把逐首程模式误放松成 aggregate |
| 缺失双单分支 | `test_solver.py::test_missing_double_branch_forces_zero_initiation` | 删除需求或错误添加不可行割 |
| 穷举战略设计 | `test_master_bounds.py::tiny`、`legal_designs` | 条件割符号、M、跨模式误用 |
| 关闭航段仍需定价 | `test_solver.py::test_pricing_visits_paths_on_closed_network_and_rejects_interrupted_certificate` | 定价域随当前 y 缩小 |
| 双基地空载平衡 | `test_solver.py::test_two_base_repositioning_is_generated_and_balances_service` | 省掉调位列 |
| 任务精确总量及首程执行 | `tests/unit/test_aggregate_validation.py` | 仅保留边际量、丢掉 pair total 或 head 关系 |
| 增量加列与中断 | `test_incremental_cg.py`、`test_dual_status.py` | 矩阵索引过期、未定价即发割 |

`route_tradeoff_data` 有一个特别有用的边界：不同首程支持不同续单，固定 `rho=.5` 时 aggregate 可行而逐首程模式不可行。这个额外固定 rho 的诊断子问题应继续区分两者；它不否定原模型可令 rho0 获得全骑手补救。

穷举跨设计割时，先以小目录和现有 `tiny()` 测试为准。为避免独立最优解多解影响，不比较每个流量元素相等。对每个 anchor 保存原始仿射系数、条件化系数、M、设计位串、运营上下界、最差违反设计。

必须测试中断：模拟预算结束、停止请求、对偶无可用证书、定价未完成。允许保留经验证的 primal 和旧的有效 LB，不允许发布新条件割，也不把受限问题成本当作完整问题下界。

现有默认独立运营容差为 `5e-5`；部分单元测试使用更严格的流量或成本容差。保持原检查标准，记录误差来源，不通过放宽容差来获得等价。

## 13 运行入口与实验记录

### 13.1 现有可用入口

下面是当前已有命令。新的 conditional 开关需实现后补入命令与结果，不能现在假称已可运行。

```bash
cd /Users/songtaowang/Documents/MLND_code
make -C cpp -j2
cpp/build/mlnd_cpp --help
```

先使用小型 probe。现有构建目标包括 `fixed-design-probe`、`formulation-probe`、`benders-probe`、`pricing-probe`。对应检查脚本位于 `cpp/tests/formulation_reference.py`、`benders_reference.py`、`cg_reference.py`。

完成小实例验收后，可将以下作为 50 OD 原算法 baseline；输出目录必须唯一且不存在，现有程序拒绝覆盖：

```bash
cpp/build/mlnd_cpp \
  --input examples/tsl_2l20h50od --k 2 \
  --method benders_cg --plain \
  --seconds 600 --gap 0.005 --threads 1 --seed 43 \
  --out outputs/conditional_benders_20261022_baseline_run01
```

同设计比较使用当前 `--fixed-design` 接口，传入已保存且经验证的实际 solution 文件。对小实例分别运行全列与 CG；大实例不要求先枚举全部完整路径。不得用旧历史结果替代同一源码和同一冻结输入下的 baseline。

Python 基础正确性测试可在实施后选择性运行：

```bash
/opt/anaconda3/bin/python -m pytest \
  tests/unit/test_model.py \
  tests/unit/test_pooling.py \
  tests/unit/test_task_relaxation.py \
  tests/unit/test_aggregate_validation.py -q
```

新增的 conditional 模式测试必须单独加入，不修改旧 global-lift 测试的预期来迁就新算法。本手册交付时没有运行这些求解测试，也没有取得新算法性能结果。

### 13.2 A 和 B 两组的公平比较

| 实验组 | 需求处理 | 割 | 其他设置 |
|---|---|---|---|
| A | 当前 lifted 子问题 | 当前普通全局对偶割 | plain |
| B | 固定 alpha 数值系数，删除 eta/u/Z/Y | 条件对偶割 | 与 A 相同的 plain |

相同实例、K、流量策略、物理路径域、线程数、随机种子、时间预算、误差阈值和验证器。相同的 initial seed policy；最终生成列集合允许因搜索轨迹不同而不同，记录差异。

固定设计实验先回答“每次 SOCP 是否更小更快”；外层实验再回答“条件割是否导致更多设计评价，整体 UB/LB 是否改善”。两组各运行和认证，不能因为新算法目标值更低就把它报告为已证明更优。

### 13.3 必需记录的指标

每轮记录：完整设计与 alpha 指纹、参数计算时间及缓存命中、建模/primal/dual/pricing/验证/master 时间、变量/行/非零元/锥数、生成列数、定价完成情况、条件割 M 及其修正、anchor 证书间隙、运营 primal UB 与 certified LB、全局 UB/LB/gap、停止原因。

总体记录：不同 alpha 的数量、同 alpha 下不同 y 的评价次数、总外层迭代数、条件割数、总耗时与内存、经验证 incumbent、全局证书和源码/输入指纹。

注意区分 `catalogue_complete` 与 `pricing_complete`：隐式 CG 不物化完整目录，仍可能完成全域定价。保留现有 `support_complete`、证书适用范围和预算中断语义。

## 14 实现 agent 的交付清单

1. 独立 opt-in 模式，旧默认算法和完整模型基准保留。
2. 当前 q-kappa-share 参数计算与 compact SOCP 构造。
3. 固定分配对偶提取、alpha 常数吸收、有效 M 和条件割插入。
4. 不改变物理路径的 CG 接入及全域证书检查。
5. eta 输出恢复、旧矩阵双向映射审计与独立运营验证。
6. 小实例新旧模型等价、同模式和跨模式割、两种流量策略、边界与中断测试。
7. 同源码同输入的 A/B 结果，分别报告规模收益、单次开销、外层迭代和最终上下界。
8. 说明新增模式支持范围、未接入的加速、运行命令及复现文件。

达到这些条件后，才能判断是否值得将新模式推广到当前较大规模实验。算法试验的结论只针对现有代码模型，不同时评价或修改论文里的运营假设。
