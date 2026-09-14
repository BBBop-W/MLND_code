# Partial Benders：在主问题中保留机队与连续运营骨架

本变体将基地可用率、idle 无人机数及基地间总流量留在主问题，仅把完整物理路径的分配交给线性子问题。它求解与现有路径模型相同的整数设施设计问题。新增代码位于 `mlnd_path/partial_benders.py`；既有求解器及历史实验保持原样。

## 1. 变量与成本分工

主问题包含设施与分配二元变量 $v=(x,y,s,b)$，连续可用率 $\rho_\ell$、idle 数量 $I_\ell$，服务总流量 $T_{od\ell m}$、空程总流量 $E_{\ell m}$，以及完整路径成本下界 $\zeta$。

记 $M_{od}=h_{od}\bar\rho$。每条服务列与空程列的单位成本沿用现有模型：

$$
\widetilde c_p=c_p-c^{G}_{od}+c^N\tau_p,
\qquad
\widetilde c^E_r=c^E_r+c^N\tau^E_r.
$$

主问题目标为

$$
\min\ C^{\mathrm{fix}}(v)+C_0+c^N\sum_\ell I_\ell+\zeta.
$$

$\zeta$ 为自由变量，因为路径成本包含被替代的骑手成本，可能为负。它具有以下有效下界，因而主问题有界：

$$
\zeta\ge
\sum_{od,\ell,m}\underline c_{od\ell m}T_{od\ell m}
+\sum_{\ell\ne m}\underline c^E_{\ell m}E_{\ell m},
$$

其中每个下划线系数取相应组的全部候选列最低单位成本。无候选列的组，其总流量固定为零。

## 2. 主问题中的连续运营结构

保留原有二元选址与唯一 OD 基地对分配约束。对每个 OD 和起点基地，令 $B_{od\ell}=\sum_m b_{od\ell m}$，添加

$$
\sum_mT_{od\ell m}\le h_{od}\rho_\ell,
$$

$$
\sum_mT_{od\ell m}\ge h_{od}\rho_\ell-M_{od}(1-B_{od\ell}),
$$

$$
0\le T_{od\ell m}\le M_{od}b_{od\ell m}.
$$

整数分配下，未选中的基地对流量为零，选中的基地对流量等于 $h_{od}\rho_\ell$。这与原模型的 OD 全局流量链接完全等价。该表达同时给出更强的分数松弛：流量直接与其实际起点基地的可用率联系。

主问题还包含

$$
\sum_{od,m}T_{od\ell m}+\sum_{m\ne\ell}E_{\ell m}
=\sum_{od,m}T_{odm\ell}+\sum_{m\ne\ell}E_{m\ell},
$$

$$
\sum_{m\ne\ell}E_{\ell m}\le\bar\kappa\rho_\ell,
\qquad
0\le\rho_\ell\le\bar\rho x_\ell,
\qquad
I_\ell\ge\frac{\rho_\ell}{1-\rho_\ell}.
$$

idle 约束继续使用普通旋转二阶锥表示。空程终点基地必须开放，故另有

$$
E_{\ell m}\le\bar\kappa\bar\rho x_m.
$$

此外，主问题加入两类来自路径集合的必要条件：

1. 若某 OD 的所有候选路径均使用两个不同的 H，对任意 $h_0$，有
   $\sum_{\ell,m}T_{od\ell m}\le M_{od}\sum_{h\ne h_0}y_h$。
   该 OD 只开零个或一个 H 时，服务总流量因此为零。存在同一 H 完成两个阶段的候选列时，不加入这一组约束。
2. 若某个设施属于一个组内所有候选路径的必经换电站集合，则该组总流量受此设施开放变量控制。

这两类约束在整数设施设计下均为原路径可行性的必要条件；设施开放后，其右端为现有组流量上界。

## 3. 路径分配 LP

固定主问题点 $(\hat v,\hat T,\hat E)$。子问题变量只有完整服务列流量 $\theta_p$ 和空程列流量 $\phi_r$：

$$
\min\sum_p\widetilde c_p\theta_p+\sum_r\widetilde c^E_r\phi_r.
$$

各组满足精确分解：

$$
\sum_{p\in\mathcal P_{od}^{\ell m}}\theta_p=\hat T_{od\ell m},
\qquad
\sum_{r\in\mathcal R_{\ell m}}\phi_r=\hat E_{\ell m}.
$$

H 与换电站激活约束保持完整列的实际 incidence 与换电访问次数。代码还保留按 OD 和空程起点分解的激活上界，从而加强对偶割。所有组流量与设施开放值只出现在 LP 的右端。

用 $D_S,D_E$ 表示组分解矩阵，用 $A_S,A_E$ 表示设施激活矩阵，用 $R\hat v$ 表示激活右端，则子问题可写成

$$
D_S\theta=\hat T,\quad D_E\phi=\hat E,
\quad A_S\theta+A_E\phi\le R\hat v,
\quad\theta,\phi\ge0.
$$

## 4. 固定连续运营量后不可行：Phase I

固定 $\rho,T,E$ 后，选址可能无法承载这些流量。对每条组分解等式添加非负未分解流量 $u$：

$$
D_S\theta+u^S=\hat T,\qquad
D_E\phi+u^E=\hat E.
$$

保持设施激活约束，求解

$$
\min\mathbf 1^\top u^S+\mathbf 1^\top u^E.
$$

该问题始终可行：取全部实际路径流量为零，$u^S=\hat T,u^E=\hat E$。

以 $\alpha^S,\alpha^E$ 表示组等式价格，以 $\beta\le0$ 表示激活行价格。Phase-I 对偶还满足 $\alpha^S,\alpha^E\le1$。其服务列检验数为

$$
\bar c^I_p=-\alpha^S_{g(p)}-A_{S,p}^{\top}\beta,
$$

空程列同理。完成全部候选列的定价后，若 Phase-I 最优值为正，就得到可行性割

$$
(\alpha^S)^\top T+(\alpha^E)^\top E+\beta^\top Rv\le0.
$$

若最优值为零，将全部人工流量上界固定为零，再把实际路径列成本切换为 $\widetilde c,\widetilde c^E$，进入真实成本 LP。

## 5. 最优性割与全候选定价

真实成本 LP 中，服务列检验数为

$$
\bar c_p=\widetilde c_p-\alpha^S_{g(p)}-A_{S,p}^{\top}\beta,
$$

空程列使用 $\widetilde c^E_r,\alpha^E$。定价始终扫描完整候选集合，包括当前关闭设施涉及的路径。每轮按服务组和空程组分别加入一条最负检验数列，直到全部检验数达到容差。

完成定价后，返回最优性割

$$
\zeta\ge(\alpha^S)^\top T+(\alpha^E)^\top E+\beta^\top Rv.
$$

这些对偶约束与主问题设计无关，因而两类割均对全部主问题方案有效。

### 数值证书

首先裁去错误符号的激活价格；Phase I 还把组价格限制在 1 以下。随后重新计算完整候选集合的检验数。若仍有微小负值，以

$$
\Delta=
\sum_{od}M_{od}\max\{0,-\min_{p\text{ 属于 }od}\bar c_p\}
+\sum_\ell\bar\kappa\bar\rho
\max\{0,-\min_{r\text{ 起于 }\ell}\bar c_r\}
$$

下调割的常数项，并记录额外浮点余量。该修正使用对所有主问题可行方案成立的总流量上界。Phase-I 割只有在修正后的对偶值仍严格为正时才用于排除当前方案；最优性割还检查修正后的原对偶间隙与路径原始残差。

## 6. 实现与计时约定

实现保留一个持久 LP，使用 dual simplex 与线性行 `Pi`。新列通过 `Column` 增量加入。切换主问题点时只更新 LP 右端；切换 Phase I／真实成本时只更新列成本和人工变量上界。无需在每次评估中重建 SOCP，也无需独立求解锥对偶。

主问题采用整数 incumbent 回调分离。每次评估都使用当前 $T,E$，不按二元选址缓存子问题结果，因为同一选址下连续运营量可能不同。

只有实际路径分配通过原始模型残差核验后，才更新整数上界。限制列 LP 的可行方案也可更新上界，但完整定价未结束时不回传新的 Benders 割。超时返回已经验证的可行方案和主问题全局下界；所有选址关闭、全部使用普通骑手的方案始终提供初始可行上界。

两个可比较实现共用完全相同的主问题和路径系数：

- `use_cg=True`：从空列池开始，持续积累路径列。
- `use_cg=False` 或 `full_columns=True`：LP 初始即包含全部候选列。

记录主问题时间、LP 求解时间、全候选定价时间、Phase-I／真实成本轮数及两类割数。LP 总时间与主问题净时间分开，避免把回调中的 LP 耗时重复归入主问题。

## 7. 与完整路径模型的等价性

任一原模型整数可行解，按 OD 基地对及空程基地对汇总列流量，即得到主问题的 $T,E$。源基地链接、平衡、时钟及新增必要条件均成立，原列流量构成子问题可行解，目标不变。

反之，主问题整数方案若具有可行路径分解，则各组等式恢复全部真实路径流量；组来源链接给出原有 OD 接单关系，主问题平衡和时钟恢复原网络约束，LP 激活行恢复设施可行性。idle 锥与完整列成本保持原样，因此得到原模型同目标值的可行解。两种形式具有相同的整数最优值。

在当前没有真实 H／换电站容量上限的版本中，固定整数设施与组流量后，每个开放组的最低成本路径还可提供一个独立核验：最优分配成本等于各组流量乘其最低开放列成本之和；没有任何开放可行列的组，其流量必须为零。正式求解器仍通过 LP 与完整定价获取可用于全部方案的对偶割。
