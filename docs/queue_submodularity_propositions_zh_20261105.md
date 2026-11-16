# 任务独立指派下的次模有效不等式与精确分离

本稿对应当前任务级独立指派模型。令 $\mathcal T$ 为固定任务类型集合，包含单订单任务 $(r)$ 和允许的有序双订单任务 $(r,s)$；K1 仅含前者。每个任务 $q\in\mathcal T$ 独立选择出发基地，指派变量为 $\alpha_{ql}\in\{0,1\}$，满足 $\sum_l\alpha_{ql}\le1$。同基地共享可用率 $\rho_l$，但同一首单的不同任务不共享指派变量。任务路径流量满足

$$
\sum_{p\in\mathcal P_{ql}}f_p=\nu_q\alpha_{ql}\rho_l,
\qquad q\in\mathcal T,\ l\in\mathcal L. \tag{A}
$$

其中 $\nu_q$ 是固定任务率，$\mathcal P_{ql}$ 遍及该任务从 $l$ 出发的全部可行路径和返回基地。正率任务没有路径时仍保留式 (A)，故 $\alpha_{ql}\rho_l=0$。这不直接禁用具有相同首单的其他任务。

令 $n_{rq}$ 为任务 $q$ 包含 OD $r$ 的次数；$(r,r)$ 取 2。任务率保持订单均值守恒 $\lambda_r=\sum_q n_{rq}\nu_q$，OD $r$ 的无人机服务量为 $\sum_{q,l}n_{rq}\nu_q\alpha_{ql}\rho_l$。需求重构原有的排序、残余机会概率、共同出发支持过滤及窗口规则均未改变；共同支持过滤只用于生成固定任务率，不是执行阶段的共同指派约束。

路径可行性按每次订单自身请求时点检查“取件时间、载货飞行时间与骑手送到家平均时间”之和。送到家时间与混合收件、混合移除时间分别保存；后两者的原有定义及柜格占用关系保持不变。以下所有路径最小值必须使用这一当前完整物理域；旧时限域的系数或证书不能直接沿用。

记 $\boldsymbol d$ 为整数网络设计，$u$ 为其可行运营方案，$c_{\rm op}(\boldsymbol d,u)$ 为实际运营成本，$Q(\boldsymbol d)=\inf_u c_{\rm op}(\boldsymbol d,u)$。以 $\zeta$ 表示运营成本上图变量，以 $C^{\rm R}=\sum_r\lambda_rc_r^{\rm R}$ 表示全部订单由骑手配送的成本，以 $F>0$ 表示单位机队成本。假设其余路径与空载运营成本非负，基地闲置机队满足 $I_l\ge\rho_l/(1-\rho_l)$。

**命题 1（逐任务节省与基地队列下界的次模性）** 对正率且支持非空的任务/基地组合，令

$$
g_q=\sum_r n_{rq}c_r^{\rm R},\qquad
S_{ql}=\left[\nu_q\left(g_q-
\min_{p\in\mathcal P_{ql}}\{c_p+FT_p\}\right)\right]_+,
\qquad [a]_+=\max\{a,0\}.
$$

这里 $c_p$ 为路径运营成本，$T_p$ 为其忙碌机队占用时间。对零率任务或正率而无路径的任务，置 $S_{ql}=0$。正部在每个任务上独立计算，不能先按共同首单聚合再取正部。对实数 $s$ 定义

$$
\Phi_F(s)=\sup_{0\le\rho<1}
\left\{s\rho-F\frac{\rho}{1-\rho}\right\}
=\begin{cases}
0,&s\le F,\\
(\sqrt{s}-\sqrt F)^2,&s>F,
\end{cases}
\qquad
f_l(A)=-\Phi_F\!\left(\sum_{q\in A}S_{ql}\right),\quad A\subseteq\mathcal T.
$$

令 $A_l(\boldsymbol d)=\{q:\alpha_{ql}=1\}$。则 $f_l(\varnothing)=0$，$f_l$ 次模，且

$$
Q(\boldsymbol d)\ \ge\ C^{\rm R}+\sum_l f_l(A_l(\boldsymbol d)). \tag{1}
$$

**证明。** 固定任一整数设计及任一可行运营方案。由订单守恒，从骑手基准中扣除各服务任务替代的骑手成本，并计入路径运营与忙碌机队成本。用完整路径域上的最小 $c_p+FT_p$ 下界化服务成本，再将每个任务净节约系数取正部，所得节约不超过 $S_{ql}\alpha_{ql}\rho_l$。无路径的正率任务由式 (A) 强制实际执行量为零，上界仍成立。丢弃非负空载成本与额外闲置库存后，得到

$$
c_{\rm op}(\boldsymbol d,u)\ \ge\
C^{\rm R}+\sum_l\left[
F\frac{\rho_l}{1-\rho_l}-\rho_l\sum_qS_{ql}\alpha_{ql}\right].
$$

逐基地在 $0\le\rho_l<1$ 上最小化右端，再对左端可行运营方案取下确界，即得式 (1)。即便某个缺支持任务使原方案的 $\rho_l=0$，放宽该限制再最小化也只会降低右端。最大化 $\Phi_F$ 的定义式，在 $s\le F$ 时取 $\rho=0$，在 $s>F$ 时取 $\rho=1-\sqrt{F/s}$，得到所列闭式。

$\Phi_F$ 是非负斜率仿射函数族的上确界，因而凸且非减。对任意 $w\ge0$，$\Phi_F(s+w)-\Phi_F(s)$ 随 $s$ 非减。因此，若 $A\subseteq B$ 且 $i\notin B$，则

$$
f_l(A\cup\{i\})-f_l(A)\ \ge\
f_l(B\cup\{i\})-f_l(B).
$$

这正是次模性。由 $\Phi_F(0)=0$ 得归一化条件。$\square$

**命题 2（排列不等式的有效性）** 对每个基地 $l$ 任取任务集合 $\mathcal T$ 的一个排列 $\pi_l$。令 $n=|\mathcal T|$，$P_{l,0}=\varnothing$，$P_{l,j}=\{\pi_l(1),\ldots,\pi_l(j)\}$，并定义

$$
m^l_{\pi_l(j)}=f_l(P_{l,j})-f_l(P_{l,j-1}),\qquad j=1,\ldots,n.
$$

则下列不等式对原整数模型的运营成本上图及其凸包有效：

$$
\boxed{\ \zeta\ \ge\ C^{\rm R}+\sum_l\sum_{q\in\mathcal T}m^l_q\alpha_{ql}.\ } \tag{2}
$$

**证明。** 固定基地并省略下标 $l$。任取实际选中任务集 $T\subseteq\mathcal T$。当 $i=\pi(j)\in T$ 时，$T\cap P_{j-1}\subseteq P_{j-1}$，次模性给出

$$
f(T\cap P_j)-f(T\cap P_{j-1})\ \ge\ f(P_j)-f(P_{j-1})=m_i.
$$

对 $i\in T$ 的各步求和，左端首尾相消，故 $\sum_{i\in T}m_i\le f(T)-f(\varnothing)=f(T)$。逐基地取 $T=A_l(\boldsymbol d)$，结合式 (1) 及 $\zeta\ge Q(\boldsymbol d)$ 即得式 (2)。线性不等式在全部整数上图点成立，因此也在其任意凸组合上成立。$\square$

**命题 3（排序给出固定排列割族的精确分离）** 给定分数点 $(\bar{\boldsymbol d},\bar\zeta)$，且 $\bar\alpha_{ql}\in[0,1]$。对于固定的集合函数 $f_l$，逐基地将 $\bar\alpha_{ql}$ 非增排序，按命题 2 计算边际系数。所得不等式在该点达到所有基地排列组合中的最大右端，也即最大带符号违反量。

**证明。** 对一个基地，记 $x_q=\bar\alpha_{ql}$，取降序排列 $x_{\pi(1)}\ge\cdots\ge x_{\pi(n)}\ge0$，置 $x_{\pi(n+1)}=0$，以下 $P_j$ 均指此排列前缀。任取另一排列产生的系数 $a$，记 $a(T)=\sum_{i\in T}a_i$。命题 2 的证明给出对所有集合均有 $a(T)\le f(T)$，特别有 $a(P_j)\le f(P_j)$。分部求和得

$$
a^\top x
=\sum_{j=1}^n\big(x_{\pi(j)}-x_{\pi(j+1)}\big)a(P_j)
\le\sum_{j=1}^n\big(x_{\pi(j)}-x_{\pi(j+1)}\big)f(P_j). \tag{3}
$$

各权重非负，不等式方向成立。对降序排列自己的系数 $m$，前缀求和恰有 $m(P_j)=f(P_j)-f(\varnothing)=f(P_j)$，所以式 (3) 在全部前缀同时取等，即 $m^\top x\ge a^\top x$。各基地的排列可独立选择，故逐基地最大化即最大化整个右端。相同分数值可任意打破平局。权重预先给定后，排序与一次前缀累加的复杂度为 $O(|\mathcal L|\,n\log n)$。$\square$

该最强性限定于**给定系数、给定分数点与所有排列构成的割族**。它不包含资源价格的选择，也不宣称得到原模型所有有效不等式中的最强一条。数学命题采用精确算术；实现对增量与截距作向有效方向的舍入保护。

**命题 4（含符号收益和缺支持任务的资源价格割）** 固定一组资源价格 $\lambda$：柜容量、航线负载和逐任务归一化航线门控采用非负乘子，基地流量平衡采用自由乘子。设 $U_l=\{q:\nu_q>0,\ \mathcal P_{ql}=\varnothing\}$。对每个 $q\notin U_l$，用 $B_{ql}$ 表示该单个任务在完整路径集上的最小调整成本的相反数，其中计入任务率、骑手替代成本、忙碌机队成本与资源价格；零率任务取 0。记 $K_\lambda(\boldsymbol d)$ 为骑手基准、资源右端价格项及空载修正之和，它是设计变量的仿射函数。

具体地，记柜负荷价格为 $\mu_i\ge0$，航线负荷价格为 $\eta_e\ge0$，基地平衡势为 $\beta_l\in\mathbb R$，逐任务归一化门控价格为 $\gamma_{qle}\ge0$。对正率且支持非空的任务，可取

$$
B_{ql}=-\min_{p\in\mathcal P_{ql}}\left\{
\nu_q\left(c_p+FT_p-g_q+\sum_i\mu_i b_{ip}
+\sum_e\eta_e n_{ep}+\beta_l-\beta_{\operatorname{ret}(p)}\right)
+\sum_e\gamma_{qle}\mathbf1_{\{e\in p\}}\right\}.
$$

其中 $b_{ip}$ 为柜格负荷，$n_{ep}$ 为航线通过次数。归一化门控使用是否经过航线的指示量，重复经过同一航线仍只计一次；通常的航线负荷项则计完整次数。相应的设计仿射项为

$$
K_\lambda(\boldsymbol d)=C^{\rm R}
-\sum_i\mu_i\operatorname{cap}_i(\boldsymbol d)
-\sum_e\eta_eU_ey_e-\sum_{q,l,e}\gamma_{qle}y_e
+\sum_a\min\{0,c'_a\}M_a(\boldsymbol d),
$$

其中 $c'_a$ 是含忙碌机队、航线价格及平衡势的空载调整单位成本。未启用的价格项取零；以上表达先逐任务求最小值，不跨同首单任务合并。

上述调整在每个整数可行方案上给出

$$
c_{\rm op}(\boldsymbol d,u)\ge K_\lambda(\boldsymbol d)
+\sum_l\left[F\frac{\rho_l}{1-\rho_l}
-\rho_l\sum_{q\notin U_l}B_{ql}\alpha_{ql}\right]. \tag{4}
$$

对支持任务令 $w_{ql}=|B_{ql}|$，$b_l=\sum_{q\notin U_l:B_{ql}<0}B_{ql}$。当 $B_{ql}\ge0$ 时令 $t_{ql}=\alpha_{ql}$，当 $B_{ql}<0$ 时令 $t_{ql}=1-\alpha_{ql}$。对 $q\in U_l$ 令 $t_{ql}=1-\alpha_{ql}$、$w_{ql}=0$。定义

$$
f_l^\lambda(T)=-\mathbf1_{\{U_l\subseteq T\}}
\Phi_F\!\left(b_l+\sum_{q\in T\setminus U_l}w_{ql}\right),
\qquad T\subseteq\mathcal T. \tag{5}
$$

则 $f_l^\lambda$ 归一化且次模，命题 2、3 可应用于变量 $t$，得到

$$
\boxed{\ \zeta\ge K_\lambda(\boldsymbol d)
+\sum_l\sum_q m^{l,\lambda}_q t_{ql}.\ } \tag{6}
$$

在给定分数点，按**变换后的 $\bar t_{ql}$** 非增排序，即得到固定价格对应的最强排列割。

**证明。** 对固定整数设计，向目标加入“资源使用量减资源右端”与非负乘子的乘积，在可行方案上非正；流量平衡价格项为零。逐任务归一化门控形如 $\sum_{p\in\mathcal P_{ql}:e\in p}f_p/\nu_q\le y_e$：$y_e=0$ 时路径流为零，$y_e=1$ 时左端不超过 $\alpha_{ql}\rho_l\le1$。因此门控对整数设计有效，不能据此称其对未加强的原分数松弛也有效。

逐任务以完整路径最小调整成本下界化，利用式 (A) 即得服务部分。若空载路径有负调整单位成本 $c'_a$，用原模型有效上界 $e_a\le M_a(\boldsymbol d)$，将 $\min\{0,c'_a\}M_a(\boldsymbol d)$ 计入 $K_\lambda$；直接基地间调位可使用总订单率乘对应航线开设变量。计入基地闲置库存下界，得到式 (4)。所有路径最小值必须针对当前完整域重新计算，不允许使用受限列池最小值替代。

若某个 $q\in U_l$ 被指派，式 (A) 强制 $\rho_l=0$，则式 (4) 中该基地括号为零；此时 $U_l\nsubseteq\{q:t_{ql}=1\}$，式 (5) 亦为零。否则全部缺支持任务均未指派，并有

$$
\sum_{q\notin U_l}B_{ql}\alpha_{ql}
=b_l+\sum_{q\notin U_l}w_{ql}t_{ql}.
$$

对 $\rho_l$ 最小化后得到式 (5)，从而建立运营成本下界。

再证明次模性。$\Phi_F$ 在整个实数域非负、凸且非减。因此未乘门控指示项的函数具有非负、随集合扩大而不减的边际。对普通元素，必要元素尚未齐备时其边际为零，齐备后为上述非减边际。对必要元素，加入后仍未补齐全部必要元素时边际为零；恰好补齐时，边际等于整个非负函数值，并随其他普通元素增多而非减。因此式 (5) 取负后的函数超模，$f_l^\lambda$ 次模。若 $U_l\ne\varnothing$，空集门控为零；若 $U_l=\varnothing$，由 $b_l\le0$ 与 $\Phi_F(b_l)=0$ 得 $f_l^\lambda(\varnothing)=0$。应用命题 2、3 即得式 (6) 和排序最强性。$\square$

**实现与其他加强的关系。** 支持规范化逐任务使用 $\alpha_{ql}\le\sum_{e\in B}y_e$，其中 $B$ 击中该任务全部路径；它通过删除零执行的冗余指派保持最优值，不对每一种原始指派编码都有效。资源价格成本割与命题 1–4 则对原整数运营上图及其凸包有效，两类作用域不能混同。

当前 L2 联合队列分离选择两个独立任务，各有“不指派、基地 0、基地 1”三态，在九个组合上验证仿射下界后对其他任务作保守提升；三态设计面则固定一个任务，使用其三个完整域认证总成本下界组装覆盖全部整数设计的割。它们不改变命题 3 的限定范围，也不允许把单一设计面的下界直接计作原问题全局下界。

完整域定价的任务修正同样利用独立指派：若 $m_{ql}\le0$ 是任务/基地遗漏路径的约化成本下界，则服务修正可取 $\sum_q\nu_q\max_l[-m_{ql}]_+$。先按共同首单合并任务再对基地取最大会低估独立基地选择下的潜在损失，不能沿用。任务航线通过量上界采用相同的“逐任务取最大、再求和”顺序。

一般的次模上图与扩展多拟阵不等式可参见 [Atamtürk 与 Narayanan (2008), §2](https://doi.org/10.1016/j.orl.2008.04.006)。本稿明确证明其用于式 (1) 和式 (5) 时所需的归一化、次模性、整数有效性与排序精确分离条件。数学有效性不等同于对生产规模加速效果的实验证明。
