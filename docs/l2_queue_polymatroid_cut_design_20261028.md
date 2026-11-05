# L2 下界割设计：资源定价与共享可用率的多面体割

日期：2026-10-28（Asia/Shanghai）。状态：主方案已实现，独立有效性检查、生产L2六次配对运行及割证书审计全部完成；三例认证LB均提高，结果见 [实现与比较报告](l2_queue_polymatroid_results_20261028.md)。本轮按用户后续授权实现并测试新旧，仍取消消融与固定主问题快照实验。生产适用目标为既有 L2：2 基地、20 柜点、50 OD。

本稿沿用 preprocessed/origin/aggregate 固定任务、完整物理路径目录、现有柜容量和队列模型；不改变普通 SOCP 的原生对偶来源。主问题只增加线性行，不增加运营变量。

## 1. 相对于现有实现的新内容

`task_savings.cpp:93-103` 已消去共享可用率，但随后用一条统一比例弦界线性化基地总收益。`airline_resources.cpp:173-275` 已保留共同 rho、分支流、柜容量、航线门控、基地平衡和空载调位，其队列用九个固定支撑点的 LP 松弛。

新构造把“每个首单类型是否分配到某基地”的二元结构用于基地队列值函数的凸化，生成一族可贪心分离的 extended polymatroid inequalities（EPI）。扩展版本再将路径、柜容量、航线及分支门控的资源价格带入该值函数。它不是新增一个普通对偶恢复 LP，也不是把一个内点的 Q 值直接当全局 LB。

已有队列弦推导见 [task_cost_strengthening_validity_20261027.md](task_cost_strengthening_validity_20261027.md)；现有完整路径资源松弛见 [airline_resource_validity_20261027.md](airline_resource_validity_20261027.md)。

## 2. 基础构造：精确凸化二元分配下的队列收益

记 F>0 为单位机队成本，C0 为全部骑手成本。对首单 r、基地 l，使用现有完整目录系数 S[r,l]>=0：它把 r 的所有正率单单/双单分支作为一个整体计价，包含路径成本与忙碌机队成本；无路径组按现有安全规则取零。不得换成只计首单自身的 lambda[r]。

原整数设计中 v[r,l]=alpha[r,l]*rho[l]，故现有推导已经给出

\[
Q(x)\ge C_0+\sum_l\left[F\frac{\rho_l}{1-\rho_l}
-\rho_l\sum_r S_{rl}\alpha_{rl}\right].
\]

定义（以下对全部实数 s 都有定义）

\[
\Phi_F(s)=\sup_{0\le\rho<1}\left\{s\rho-F\frac{\rho}{1-\rho}\right\}
=\begin{cases}0,&s\le F,\\(\sqrt{s}-\sqrt F)^2,&s>F.\end{cases}
\]

这是凸、非减函数。于是 Q(x)>=C0-sum_l Phi_F(sum_r S[r,l]*alpha[r,l])。该右端关于 alpha 为凹函数，不能用其切线作下界。

对每个基地定义集合函数

\[
f_l(A)=-\Phi_F\left(\sum_{r\in A}S_{rl}\right).
\]

由于 -Phi_F 凹、S 非负，f_l 满足边际递减，即次模性；它可以是负值、非单调增加，不影响 extended polymatroid 结论。任取排列 pi_l，令 A_j 为排列前 j 项，A_0=empty，定义

\[
\Delta_{\pi_l(j),l}=f_l(A_j)-f_l(A_{j-1}).
\]

得到原整数模型及其凸包上的有效线性割

\[
\boxed{\theta\ge C_0+\sum_{l=1}^{2}\sum_{j=1}^{R}
\Delta_{\pi_l(j),l}\alpha_{\pi_l(j),l}.}
\tag{EPI-0}
\]

**证明。** 次模集合函数的前缀边际向量属于其 extended polymatroid，因此对任意二元向量 alpha，其线性值不超过 f_l 对应集合的值；两基地相加再使用上述 Q 下界即可。线性不等式在所有整数可行点成立，故也在其凸包成立。该证明不声称它是旧连续 McCormick SOCP 的冗余约束。

**分离。** 在目标分配 alpha_bar 上，各基地分别按 alpha_bar[r,l] 降序排列，即得该基地此族中最强的一条。各基地独立最大化后相加即为 EPI-0 的最强分离。整数点把取1项排在取0项之前，得到该点的精确解析队列界。复杂度为 O(L R log R)，L2 仅两次50项排序与每基地51个闭式函数值，不枚举2^50个集合。

理论依据：Atamtürk–Narayanan (2008), §2, Proposition 1及其后的贪心极点/分离结论，适用于一般次模函数，见 [原文](https://atamturk.ieor.berkeley.edu/pubs/_published/orl36-2-2008.pdf)。SOCP中利用二元限制的次模结构构造凸包不等式，也见 [Atamtürk–Gómez (2020)](https://optimization-online.org/wp-content/uploads/2018/12/7003.pdf)。本稿的具体队列值函数及资源定价推导是针对当前代码的构造，不是对上述论文现成公式的直接照搬。

## 3. 为什么能比已有弦界更强

现有静态界令 Sbar_l=sum_r S[r,l]、eta_l=Phi_F(Sbar_l)/Sbar_l，然后使用 theta>=C0-sum eta_l*S[r,l]*alpha[r,l]。

凸性给出 Phi_F(s)<=eta_l*s，故对每个整数集合，精确集合值 f_l 不低于旧线性弦界。EPI族的最大值是该集合函数在单位立方体上的凸包下包络，因此整个 EPI 族不弱于这条弦界，包括分数目标点。单条排列割不必处处支配旧弦界，实施时保留旧行。

纯代数示例（不是新增算例运行）：一个基地有两个首单组，F=1，S1=S2=4。旧弦界在 alpha=(1,0) 给 theta>=C0-1.671573。排列(1,2)给出

\[
\theta\ge C_0-\alpha_1-2.343146\alpha_2,
\]

在同一整数设计得到 C0-1，严格提高0.671573。反向排列覆盖(0,1)。这证明新族可以加强整数设计处的成本近似，并非只帮助MIP的分数节点。

同一简化结构在 alpha=(1/2,0) 时，原连续 McCormick 可取 rho=v1=1/2，运营下界为 C0-1；EPI要求 C0-1/2。因此它还可以强于仅靠该连续值函数的全部普通支持平面。这个代数见证不代表生产L2的实际LB增量。

## 4. 加入资源价格，使割同时约束容量和航线

只用 EPI-0 仍放松了大量网络资源。扩展版本令 mu_i>=0、eta_e>=0 分别定价柜容量行和真实航线负载行，lambda_l 为自由基地平衡势。令 tau[k,l,e]>=0 定价已有的归一化分支航线行

\[
\sum_{p\in P_{kl}:e\in E(p)}q_p\le y_e,
\qquad q_p=\psi_p/\nu_k.
\]

该行对整数航线有效；其中经过同一弧多次的任务在这个门控中只计“是否经过”，在物理负载行中仍按真实重数 n[e,p] 计数。

对每个正率分支 k 与路径 p 定义归一化流量的调整成本

\[
A_{klp}=\nu_k\left[c_p+Ft_p-g_k+
\sum_i\mu_i\ell_{ip}+\sum_e\eta_e n_{ep}
+\lambda_{o(p)}-\lambda_{d(p)}\right]
+\sum_{e\in E(p)}\tau_{kle}.
\]

g_k 是任务中所有订单的骑手成本，AA计两次。这样写不需要用tau除以很小的正任务率。定义

\[
B_{rl}=-\sum_{k:\,head(k)=r}\min_{p\in P_{kl}}A_{klp}.
\]

最小值必须扫描完整物理目录，包括所有返回基地；不能按当前开放航线筛路径，也不能只扫描RMP列。

记 C_i(z)=sum_s occupancy_limit[i,s]*z[i,s]，U_e 为当前原模型实际采用的物理航线界。L2直接空载路径 a 的调整成本为

\[
d_a=c_a+Ft_a+\sum_e\eta_e n_{ea}+\lambda_{o(a)}-\lambda_{d(a)}.
\]

原模型在二元航线下满足 0<=empty_flow[a]<=D*y[e(a)]，D为原订单总率。因此即使d_a<0，其最小贡献也有安全界 D*min(0,d_a)*y[e(a)]。令

\[
K(z,y)=C_0-\sum_i\mu_i C_i(z)-\sum_e\eta_e U_e y_e
-\sum_{k,l,e}\tau_{kle}y_e
+D\sum_{a\in P^0}\min(0,d_a)y_{e(a)}.
\]

则对每个原整数设计

\[
\boxed{Q(x)\ge K(z,y)-\sum_l\Phi_F\left(\sum_r B_{rl}\alpha_{rl}\right).}
\tag{PRICE}
\]

**证明。** 将需求守恒代入目标，得到全部骑手基线减任务骑手替代成本，加路径成本和机队成本。`fleet_use`使机队成本至少覆盖忙碌时间与各基地idle。对每个原可行解加入非正的容量、航线、门控拉格朗日项以及为零的平衡项，不增加其目标。各分支按其完整目录最小调整成本下界化；分支流量仍为nu_k*alpha[r,l]*rho[l]。empty项按有限上界下界化。最后使用idle>=rho/(1-rho)并逐基地消去rho，即得PRICE。所有价格可以是任意有限的上述符号向量，不要求它们单独构成完整对偶可行解；完整目录最小值和解析最小化承担证书责任。

**无路径正率分支。** 不把空集最小值存成大M。记 U_l 为存在缺失正率分支的head集合；若某个r属于U_l且alpha[r,l]=1，原分支等式及共同可用率强制rho[l]=0。这种冗余整数编码仍可能可行，不能擅自断言alpha必须为0。一个安全但较弱的PRICE版本把这些组B置零并放松该联动；第5节给出保留该联动的首选割。

**相对现有资源LP的边界。** 仅使用mu/eta/lambda时，遗漏现有tau门控，不能宣称新价格界支配完整资源LP。即使保留tau，也应逐项对齐缺路径处理、队列、rho域和数值保护后才能声明特定价格证书的支配关系。本稿只证明PRICE有效，以及EPI-0对同系数旧静态弦界的族级支配；不预先声称支配全部既有动态资源割。

## 5. 有符号B的线性割

资源价格可能使B为负，不能直接按“非负权重的凹函数”套用次模结论。对每个基地令

\[
t_{rl}=\begin{cases}\alpha_{rl},&B_{rl}\ge0,\\1-\alpha_{rl},&B_{rl}<0,\end{cases}
\quad w_{rl}=|B_{rl}|,\quad b_l=\sum_{r:B_{rl}<0}B_{rl}.
\]

先对支持完整的组 r不属于U_l 做上述变换，b_l和加权和也只包含这些组。若U_l为空，定义 f_l(A)=-Phi_F(b_l+sum_{r in A}w[r,l])。b_l<=0且Phi_F(b_l)=0，仍是归一化次模函数。

对U_l非空的情形，另令t[r,l]=1-alpha[r,l]（r属于U_l），把这些位作为零权重的必要门控。使用

\[
f_l(A)=-\mathbf{1}\{U_l\subseteq A\}\,
\Phi_F\left(b_l+\sum_{r\in A\setminus U_l}w_{rl}\right).
\]

它精确保留“任何无路径组被分配，则该基地rho=0”的队列最小值。Phi作用于非负权重和是非负、非减、超模集合函数；再乘所有必要门控齐备的AND指标仍为超模：普通元素的增益在门控齐备时才出现，门控元素的增益为零或完整Phi值，并随其余集合增长而不减。取负后f_l为次模，且f_l(empty)=0。因此仍可对同一个50元素集合直接贪心分离，不需要大M或新增变量。

按目标t_bar降序构造前缀边际Delta，得到

\[
\boxed{\theta\ge K(z,y)+\sum_{l,r}\Delta_{rl}t_{rl}.}
\tag{EPI-P}
\]

将t=1-alpha展开后仍是现有战略变量与theta上的一条线性行。若采用更简单的B+=max(B,0)，也有效，但会丢掉负收益组与共同rho的成本联动；有符号版本避免该损失。

## 6. L2分离策略

1. 零资源价格使用现有S+生成EPI-0，系数准备只做一次。
2. 本版从现有航线资源LP的已验证对偶读取资源价格候选；mu/eta/tau按本稿最小化约定取非负，lambda保留自由符号。普通SOCP原生对偶作为候选来源尚未接入。没有tau的候选仍可用，但不宣称支配旧资源LP。
3. 对每个新价格向量完整扫描路径，计算并缓存B、K及空载修正。50个OD的所有正分支均保留。L2只有两个基地，平衡势可固定一个为零，只剩一个势差。
4. 在整数有效强化主问题的分数解或整数候选上，对每个缓存向量做两次50项排序，精确分离该固定价格下的EPI族。检查的是新行对当前master(theta,x)的违反量，而不是求完普通recourse以后再与旧theta比较。
5. 每轮保留一条最违反且系数不重复的行；已有静态弦、普通Benders及资源割全部保留。已实现最多8个缓存profile，始终保留EPI-0，其余按时间更新；每个新profile最多1秒，完整扫描未完成则原子拒绝。全局最多1000条新行。这些是工程预算，并非已证最优选择。
6. 便宜分离可放在必要路径支持检查之后、昂贵运营recourse之前；成本割重启次数必须有界，保留现有完整recourse机会，避免结构割让整数CG长期得不到执行。
7. 这些行进入原整数主问题/明确标注的整数有效根松弛。算法的全局LB仍取该主问题的认证bound，不把PRICE值、单点Q值或违反量直接当全局LB。

固定价格的分离是精确的；价格向量搜索只建议采用已有证书池和少量更新，不声称已经求得所有价格与所有排列中的最强割。

## 7. 数值与实施边界

- 保留生产认证容差。正任务率严格按存储值>0处理，不删除小正率。
- 完整目录成本最小值用向下保护得到B的安全上界；Phi非减，使用该B不会提高PRICE到不安全方向。
- s>F且接近F时，可用 (s-F)^2/(sqrt(s)+sqrt(F))^2 计算Phi，降低相减损失。
- 前缀差、补变量展开、容量/航线常数合并需统一保守误差界；对[0,1]设计位，一个充分补偿是从截距扣除常数误差与各系数绝对误差之和。
- 固定价格下的全部路径检查、原任务守恒和整数有效性是证书条件。不能以受限列MIP best bound充当价格界。
- 完整路径数仍与目录规模有关；排序开销很小不代表首次路径扫描免费。缓存价格向量对应的B和K，以免在每个master候选重复扫描。

## 8. 第二路线与结论状态

小组部分设计面上的残差证书及首次失配提升，见 [l2_face_cut_construction_20261028.md](l2_face_cut_construction_20261028.md)。它可直接覆盖一批不同设计，但每次需要完整域SOCP界，且若参考g已在面内某点紧，残差下界必为零。因此作为有明确触发条件的第二路线，不替代首选的闭式EPI分离。

已建立：EPI-0有效性、相对旧静态弦界的族级支配与严格代数见证；PRICE的完整域有效性；负B补变量处理；固定价格的greedy分离。

尚未建立：对全部现有资源割的支配、跨更多实例及seed的性能结论、最佳价格池与调用频率。本批三个L2实例的最终LB提高已实测，见结果报告。没有执行消融或固定master快照实验。新机制默认关闭。

## 9. 本版实现与验证

入口为 `--queue-polymatroid-cuts`，完整算法位于 `cpp/src/queue_polymatroid.cpp`，主问题调用位于 `cpp/src/benders.cpp`。单独开启提供EPI-0；同时开启 `--airline-resource-cuts` 后，复用其资源价格生成EPI-P，`--airline-resource-balance` 进一步提供基地平衡势。分离放在强化根、必要支持检查后的整数候选和已有支持根刷新位置，不因新割跳过当前候选的完整recourse。

两基地有效性probe已实际通过：K1/K2各11个完整固定设计SOCP、10个整数epigraph凸组合，每类1323次跨点核对；signed B、unsupported AND、负调价空载、完整目录与零预算保护均通过。完整集成probe验证新割实际插入，主问题无新增运营变量，认证LB未越过独立完整模型可行UB。结果保存于 `outputs/l2_queue_epi_20261028/validity.json` 和 `integration_final.json`。有限点检查用于排错；全域有效性依据本稿推导及完整路径证书。

生产比较协议为 `outputs/l2_queue_epi_20261028/PROTOCOL.md`，只比较同一新二进制的开关关闭/开启，M01/M07/M08各600秒、seed43、单线程；六次均已完成并验证，147条新割均通过完整目录独立重算。最终指标见 `outputs/l2_queue_epi_20261028/comparison/analysis/REPORT.md`。第二路线设计面残差割尚未实现。
