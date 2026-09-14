# 纯整数战略主问题：小规模凸内层与隐式全路径割

主问题仅包含设施选址、OD 基地对选择二元变量及成本下界变量。可用率、idle 数量和所有运营流量均放在内层。下述方法通过“当前开放网络的精确最短任务成本”和“全候选任务的成本下界”构造整数主问题割，无需枚举全部 OD–基地对–柜对–换电路径。

实现：`mlnd_path/integer_recourse.py`。它既可适配现有有限 `Instance`，也可直接接收隐式路径 oracle 的数组。

## 1. 固定整数设计后，内层只剩基地维度的凸网络问题

固定二元设计 $\hat v$。每个被选中的 OD 只有一对基地 $(\ell,m)$。令

$$
q_{od\ell m}(\hat v)
=\min_{p\text{ 全部设施开放}}
\{c_p-c^G_{od}+c^N\tau_p\},
$$

并令 $q^E_{\ell m}(\hat v)$ 为最低开放空程成本（含机队占用费）。成本已把随机任务时间的期望和末端 utility 的成本权重纳入。

将已选择的 OD 汇总为

$$
D_{\ell m}=\sum_{od:\hat b_{od\ell m}=1}h_{od},
\qquad
C_\ell=\sum_{od,m:\hat b_{od\ell m}=1}h_{od}q_{od\ell m}.
$$

当前模型没有真实柜或换电设施吞吐容量，开放约束的作用是禁止使用关闭设施。因此每个服务组和空程组可使用其最低成本开放路径。内层精确降为

$$
Q(\hat v)=C_0+\min_{\rho,I,e}
c^N\sum_\ell I_\ell+\sum_\ell C_\ell\rho_\ell
+\sum_{\ell\ne m}q^E_{\ell m}e_{\ell m},
$$

$$
\Big(\sum_mD_{\ell m}\Big)\rho_\ell-\sum_mD_{m\ell}\rho_m
+\sum_m e_{\ell m}-\sum_m e_{m\ell}=0,
$$

$$
\sum_m e_{\ell m}\le\bar\kappa\rho_\ell,
\qquad
0\le\rho_\ell\le\bar\rho\hat x_\ell,
\qquad I_\ell\ge\frac{\rho_\ell}{1-\rho_\ell}.
$$

无开放空程路径时，相关 $e_{\ell m}$ 不存在。若某个正需求 OD 已分配至 $(\ell,m)$，但没有开放服务路径，则增加 $\rho_\ell=0$；这也会影响同基地的其他 OD。零需求 OD 不产生这一限制。

取 $\rho=I=e=0$ 始终可行。因此每个战略设计都具有运营补救方案，且 $Q(\hat v)\le C_0$。缺少服务路径产生的是成本信息及零可用率结果，不产生排除整个战略设计的不可行割。

在南山 8 个基地的实例上，内层至多包含 8 个 $\rho$、8 个 $I$ 和 56 个空程流量。OD 数量进入预处理的系数 $D,C$，不进入连续求解器的行列数。

## 2. 连接原模型：来源乘积的线性表示

为生成全局整数割，使用与原整数模型等价的来源链接。记 $T_{od\ell m}$ 为完整列流量之和，$M_{od}=h_{od}\bar\rho$，则

$$
\sum_mT_{od\ell m}\le h_{od}\rho_\ell,
\quad
h_{od}\rho_\ell-\sum_mT_{od\ell m}
\le M_{od}\Big(1-\sum_mb_{od\ell m}\Big),
\quad
T_{od\ell m}\le M_{od}b_{od\ell m}.
$$

这些变量和约束仅用于内层对偶推导，不放入主问题，也无需在固定设计内层中显式建立。

对前两行取非负乘子 $a_{od\ell},b'_{od\ell}$，定义
$\alpha_{od\ell}=b'_{od\ell}-a_{od\ell}$；对最后一行取 $\gamma_{od\ell m}\ge0$。平衡价格为 $\pi_\ell$，重定位时钟价格为 $\nu_\ell\ge0$，基地可用率上界价格为 $\sigma_\ell\ge0$。

不计设施罚价时，服务列检验数为

$$
\bar c_p=\widetilde c_p-\alpha_{od\ell}
+\gamma_{od\ell m}+\pi_\ell-\pi_m.
$$

对当前选中且有开放路径的组，令

$$
\alpha_{od\ell}=q_{od\ell m}(\hat v)+\pi_\ell-\pi_m,
\qquad \gamma_{od\ell m}=0.
$$

其全开放路径检验数直接由精确开放最短路径成本保证。其他来源的 $\alpha$ 取零，并令 $b'=\max(\alpha,0)$、$a=\max(-\alpha,0)$。

缺路导致的 $\rho_\ell=0$ 在小内层中具有上界乘子。若该基地开放，把这个乘子除以一个缺路且正需求 OD 的 $h_{od}$，加到其有限的有效路径价格中，同时移除该项在普通基地上界中的乘子。这样保留生成点的对偶值，且避免把“无路径”的运营限制写成错误的战略不可行性。

## 3. 不枚举关闭路径的严格补全

先计算所有设施开放时的精确组最低成本 $\underline q_{od\ell m}$。任意路径均满足

$$
\widetilde c_p\ge\underline q_{od\ell m}.
$$

当没有组级下界时，可使用 $\widetilde c_p\ge-c^G_{od}$，其依据是原始执行成本与机队占用费非负。

对当前未选择的基地对，直接设置

$$
\gamma_{od\ell m}=
\max\{0,\alpha_{od\ell}-\pi_\ell+\pi_m-\underline q_{od\ell m}\}.
$$

这对该组所有开放及关闭路径同时有效；由于当前 $\hat b_{od\ell m}=0$，它不降低生成点的对偶值。

对当前选中的组，令有效价格为 $q^{\mathrm{eff}}_{od}$，在每个当前关闭设施上赋予该 OD 的罚价

$$
d_{od}=\max\{0,q^{\mathrm{eff}}_{od}-\underline q_{od\ell m}\}.
$$

任何未开放路径至少使用一个关闭设施，因此总罚价不少于 $d_{od}$，覆盖该路径最大的可能成本缺口。全开放路径继续由精确开放最短路径核验。罚价在生成点的右端均为零。

这里使用以下整数有效的设施激活上界：某 OD 使用设施 $k$ 的总流量不超过 $M_{od}v_k$。设施集合包括出发基地、返回基地、两个柜及实际使用的换电设施。该行只刻画开放，采用“是否使用”的 incidence，而不使用换电次数作为物理容量。

空程采用同样的方法：每个起点基地的总流量不超过 $\bar\kappa\bar\rho$，其使用某基地或换电设施的流量不超过 $\bar\kappa\bar\rho v_k$。出发基地明确包含在设施集合内，因此闭起点空程也被覆盖。若全开放空程成本下界为 $\underline q^E_{\ell m}$，起点 $\ell$ 的通用罚价可取

$$
d^E_\ell=\max_{m\ne\ell}
\{0,\pi_m-\pi_\ell-\nu_\ell-\underline q^E_{\ell m}\}.
$$

上述补全只使用组成本数组与设施开放向量。其证书覆盖隐式完整路径集合，不要求扫描或存储全部关闭路径。

## 4. 回传的二元仿射割与数值核验

令

$$
A_\ell=\sum_{od}h_{od}\alpha_{od\ell}-\bar\kappa\nu_\ell+\sigma_\ell,
\qquad
J_\ell(A)=\min_{0\le r\le\bar\rho}
\left\{c^N\frac r{1-r}+Ar\right\}.
$$

$J$ 可直接计算：当 $A\ge-c^N$ 时取 $r=0$；否则取
$r=\min\{\bar\rho,1-\sqrt{c^N/(-A)}\}$。

因此割为

$$
\zeta\ge C_0+\sum_\ell J_\ell(A_\ell)
-\sum_{od,\ell}M_{od}b'_{od\ell}
+\sum_{od,\ell,m}M_{od}(b'_{od\ell}-\gamma_{od\ell m})b_{od\ell m}
-\bar\rho\sum_\ell\sigma_\ell x_\ell
-\text{设施激活罚价项}.
$$

全部连续变量已被取下确界，右端只有战略二元变量。代码返回 `cut_B[OD,L,L]`、`cut_facility[X,Y,S]` 与截距，避免构造数十万个字符串键。

数值实现独立求解小内层的显式锥对偶；割证书使用上述解析 $J$，不依赖浮点锥向量恰好可行。全开放空程检验数的微小负值按各起点的全局流量上界修正截距。只有生成点的原对偶差达到容差时才标记内层评价完成。oracle 未完成时，不能把未知路径当作无路径并回传此类割。

## 5. 可立即加入的战略预割

### 5.1 服务收益上界

记 $G_{od\ell m}=M_{od}\max\{0,-\underline q_{od\ell m}\}$。忽略非负 idle 与空程成本得到

$$
\zeta\ge C_0-\sum_{od,\ell,m}G_{od\ell m}b_{od\ell m}.
$$

所有设施开放时仍不存在路径的组，其可获取服务收益取零。

### 5.2 分配 idle 成本的更强预割

为每个基地选择非负权重 $w_{od\ell}$，满足 $\sum_{od}w_{od\ell}\le1$。定义

$$
\Gamma_{od\ell m}=
\min_{0\le r\le\bar\rho}
\left\{h_{od}\underline q_{od\ell m}r+c^Nw_{od\ell}\frac r{1-r}\right\}.
$$

则 $\zeta\ge C_0+\sum\Gamma_{od\ell m}b_{od\ell m}$。每个 OD 至多选择一个基地对，因此分配出去的 idle 成本不会超过每个基地实际的 idle 成本。所有系数均有一维闭式解，可向量化计算。

### 5.3 设施贡献预割

对少量关键设施 $k$，再调用一次“仅关闭 $k$、其他设施全开放”的 oracle，得到 $\Gamma^{(-k)}$；无路径组的收益系数取零。设

$$
\Delta_k=\sum_{od}\max_{\ell,m}
\{0,\Gamma^{(-k)}_{od\ell m}-\Gamma_{od\ell m}\}.
$$

则

$$
\zeta\ge C_0+\sum\Gamma^{(-k)}_{od\ell m}b_{od\ell m}-\Delta_k v_k
$$

全局有效：$v_k=0$ 时使用缺少该设施的收益上界；$v_k=1$ 时，$\Delta_k$ 足以覆盖每个 OD 最多一个基地对的系数变化。这类割把关键柜或换电设施的价值直接放入战略主问题，同时保留零运营的补救方案。

吞吐或方向性失衡的信息可进一步由历史内层的平衡价格生成网络感知预割；第一轮实现优先采用上述闭式收益／idle 预割与精确完成的运营最优性割。
