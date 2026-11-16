# L2 资源模型缺口与整数队列割构造

核对日期：2026-10-28。范围：当前 preprocessed/origin/aggregate、完整物理路径域、2 基地/20 柜点/50 OD。本文是只读推导及实现候选，没有运行求解、修改算法或证明生产 LB 已提高。

## 1. 已存在的结构和真实缺口

原 SOCP 的 `rho`、`idle`、`fleet` 见 `cpp/src/formulation.cpp:308–310`；共同 `v=alpha*rho` 的 McCormick 行见 `:358–374`；固定单单/双单率见 `:411–425`。柜点容量、基地平衡和机队行见 `:453–461`，物理路径列同时进入这些行及航线门控，见 `:491–510`。航线端点设施逻辑已在战略模型 `:191–194`。

完整路径资源 LP 已保留：共同 rho 和 McCormick（`cpp/src/airline_resources.cpp:174–180`）、柜点和总航线容量（`:181–198`）、每个正率任务分支的归一化路径流等式（`:205–221`）、每分支/航线归一化 gate（`:232–241`），以及服务和空载起返基地平衡（`:199–202,222–225,246–265`）。一个路径重复经过同一航线时，归一化 gate 只计“存在”，总航线负载仍按重复次数计。不能把这些结构再次作为新方案。

生产 lower 三例的资源元数据分别包含 31,632/57,858/45,260 个归一化 gate，`balance_relocation=true`，并有 64/48/99 个无物理路径的正率分支。它们来自 `outputs/bounds_improvement_20261028/comparison/runs/tsl_L02_M{01,07,08}/double/benders_lower/attempts/001/result.json` 的 `airline_resource_metadata`。

对固定整数设计，归一化 gate 与开/关路径一致，以上资源模型已基本保留完整运营的线性结构。显式非线性缺口是：原 SOCP 用精确 `idle >= rho/(1-rho)`（开基地时，`formulation.cpp:340–356`），资源 LP 仅用 9 个固定切点 `0,.25,.5,.75,.9,.95,.97,.98,.99` 的下包络（`airline_resources.cpp:267–271`）。资源 LP 还未显式保留原机队/idle 有限上界；使用精确 queue 且取不劣于全骑手基准的运营最优值时，原非负成本给出的机队上界可恢复。因此，不能凭这几份模型宣称“缺共享 rho”“缺联合路径选择”或“缺空载平衡”。

更重要的割构造差异是：`task_savings.cpp:93–103,118–127` 只使用全部 head 总收益决定的一条比例 chord；完整资源 LP 则在线性化队列/McCormick 模型上取对偶（`airline_resources.cpp:273–281,399–415`）。二者均未对“整数 head 集合决定的精确队列净收益”构造集合函数割。

## 2. 候选割：资源定价后的精确队列集合函数割

记 `F=d.fleet_cost>0`，`C0=sum_r demand[r]*ground_cost[r]`。正率分支 `k=(r,l,s)` 的率为 `nu_k`，路径集为 `P_k`，奖励 `g_k=ground_cost[r]+1_{s>=0}ground_cost[s]`。路径实际流量 `f_p=nu_k*q_p`，分支满足 `sum_{p in P_k} f_p=nu_k*alpha_rl*rho_l`。

原运营目标不小于

```
C0 + sum_p (c_p + F*time_p - g_k) f_p
   + sum_e (c_e + F*time_e) empty_e + F*sum_l idle_l.
```

这是消去骑手守恒、用机队约束替代机队成本的直接结果（`formulation.cpp:315,453–457,498–510`）；在运营最优解处可取等。

选择非负柜点价格 `mu_h`、总航线负载价格 `eta_e`、归一化分支 gate 价格 `tau_{k,e}`，以及自由基地平衡势 `lambda_l`。归一化 gate 是 `sum_{p in P_k:e in p} f_p/nu_k <= y_e`。定义完整路径调整成本

```
a_p = c_p + F*time_p - g_k
      + sum_h mu_h*slot_time_ph
      + sum_e eta_e*arc_count_pe
      + lambda_origin(p) - lambda_return(p)
      + sum_{e present in p} tau_{k,e}/nu_k,
B_rl = -sum_{k headed by (r,l)} nu_k * min_{p in P_k} a_p.
```

最后一项按航线存在计数；不能照搬总负载的重复次数。令 `K_h(z)` 为现有容量级别 RHS，`U_e` 为同一完整路径域的现有航线流上界。空载路径的调整成本为 `a_empty=c_empty+F*time_empty+lambda_origin-lambda_return+sum_e eta_e*arc_count_e`。在 L2 当前 direct 路径域，每个方向只有一条空载路径；由 `0<=empty_e<=D*y_e`，其中 `D=sum demand`，其下界贡献为

```
E(y) = sum_empty D*min(0,a_empty)*y_empty.
```

因此不必强制所有空载调整成本非负；若直接丢弃负成本空载项，割会失效。原空载上界/门控证据见 `formulation.cpp:321,470–485` 与 `airline_resources.cpp:261–264`。

对全实数 S 定义

```
H(S) = max_{0<=rho<1} [S*rho - F*rho/(1-rho)]
     = 0                              if S<=F,
       (sqrt(S)-sqrt(F))^2            if S>F.
```

H 凸且非减。对每个原模型整数设计，有

```
Q(x) >= C0 - sum_h mu_h*K_h(z) - sum_e eta_e*U_e*y_e
            - sum_{k,e} tau_{k,e}*y_e + E(y)
            - sum_l H(sum_r B_rl*alpha_rl).                 (1)
```

证明：将有效资源行乘非负价格、平衡行乘自由势加到最小化目标，得到不大于原可行解成本的拉格朗日函数；完整分支路径最小值给出每 head 的线性 rho 系数，再独立解析最小化每基地的 idle/rho 项。原有限机队/rho 上界在此被放松，方向安全。原 `alpha<=facility` 意味着关闭基地没有正分配；忽略额外 rho 开门约束不会破坏下界。

**缺路径处理。** 正率分支没有路径时，不允许用任意有限 M 代替 `min=+infinity`。原模型迫使该 head-origin 的 v 为零；若 alpha 为一，还迫使同基地 rho 为零。令 U_l 为这类 head 集合，仅对其余 head 计算有限 B。把 (1) 中该基地的 H 项替换成

```
1{alpha_rl=0 for every r in U_l} * H(sum_{r not in U_l} B_rl*alpha_rl).
```

它精确保留缺路径导致的共享 rho 关闭逻辑，不排除原模型允许的 `alpha=1,rho=0` 冗余设计。下一节给出该乘积的 greedy 线性化。

## 3. 将 (1) 变成跨整数设计的仿射主问题割

先考虑 U_l 为空。对每个基地单独处理 signed B：若 `B_r>=0`，置 `t_r=alpha_r`；若 `B_r<0`，置 `t_r=1-alpha_r`。令 `w_r=abs(B_r)>=0`、`b0=sum_{B_r<0}B_r`，则 `sum B_r*alpha_r=b0+sum w_r*t_r`。这里 `b0<=0`，故 `H(b0)=0`。

任取 head 的排列 pi，令 `W_j=sum_{i<=j}w_pi(i)`，`delta_pi(j)=H(b0+W_j)-H(b0+W_{j-1})>=0`。对每个二元 t，

```
H(b0+sum_r w_r*t_r) <= H(b0)+sum_r delta_r*t_r.             (2)
```

证明：按排列依次加入实际被选择的元素。H 的凸性意味着该元素在“所有前序元素”基础上的增量不小于在“实际被选前序元素”基础上的增量。逐项相加即得 (2)。这等价于 `-H` 子模函数的 greedy epigraph 割。

若 U_l 非空，再为其中 head 定义 `t_r=1-alpha_r`，令

```
F_l(A) = 1{U_l is a subset of A}
         * H(b0 + sum_{r in A and r not in U_l} w_r).
```

F_l 仍是非负、单调、超模集合函数：普通 head 的边际在必需 gate 未齐备时为零，齐备后为 H 的非减边际；任一 gate head 的边际为零或 H，随着其他元素加入不会减小。因此用 `delta_pi(j)=F_l(prefix_j)-F_l(prefix_{j-1})` 同样有 `F_l(A)<=sum_{r in A}delta_r`。实现只需计数尚缺几个 gate，所有量有限，无需大 M。任意整数分离点仍可由“t=1 排前”取得该集合函数的精确值。

把 (2) 代入 (1)，再将补变量展开成原 alpha，即得只含现有 theta、alpha、容量和航线变量的仿射割。每一行对所有原整数设计有效，继而对其凸包有效。排列按分离点 t 降序可得到该 greedy 割族的最强违反行；整数点将 t=1 排在前面，因此在该点 (2) 精确。单行不保证在全部设计逐点支配旧行，应保留旧行取共同下界。

**不是只改善分数点。** F=1、同基地两个 head 的收益 B1=B2=4。旧比例 chord 的 saving 系数均为 `H(8)/2=1.671573`。以 head1 在前的新排列给系数 `H(4)=1`、`H(8)-H(4)=2.343146`。在整数设计 `(alpha1,alpha2)=(1,0)`，新割给 `Q>=C0-1`，旧割只有 `Q>=C0-1.671573`，严格提高0.671573。反序行覆盖另一个单 head 设计。该例证明割族可严格加强已有 chord，不证明生产全局 LB 的收益。

## 4. 与已有机制的区别、分离成本和证书范围

新内容是“资源价格 + 完整路径扫描 + 精确 queue 共轭 + 二元集合函数提升”。没有重新引入已实现的共享 rho、McCormick、归一化 gate 或空载平衡；没有自建普通 SOCP 证书 LP。可先复用现有资源分离的价格向量，再构造新割；价格不必最优，只须符号合法、路径域完整、数值方向保守。

若仅保留 mu/eta/lambda 而不保留 tau，仍有新有效割，但不能声称优于当前完整资源 LP，因为会丢失数万条 gate 的价格信息。保留全部资源/gate价格、缺路径零吞吐逻辑和原空载有限域后，整数设计处对剩余变量精确最小化不弱于这些相同资源价格下再使用有限 queue 切线和其他任意可行对偶乘子的旧证书。这是同价格、数学精确计算下的比较；不意味着单条新 affine 行在所有其他整数点支配全部旧割，也不保证浮点修正后仍逐位更强。若将 unsupported head 的 B 简单设零并丢弃共同 rho 关闭逻辑，该不弱性承诺不再成立。

L2 下每轮只需扫描已有完整路径目录（本批约3.6万–10.3万服务路径、2条空载路径），按每条路径的少量柜点/航线累加调整成本；再对每基地50个 head 排序并求50个 H 增量。排序是 `O(2*50*log50)`，主耗时为完整路径和已有非零系数扫描。不新增主问题变量，不需枚举全部 assignment 组合，不需再次构建数万行资源 LP。新价格的搜索成本与割构造成本须分开报告；如果重新优化资源价格，不能把该成本省略。

浮点实现必须令 B 成为真实最大 saving 系数的上界，例如对路径最小成本向下取安全界；H 非减保证这一步只会减弱割。delta 及常数也应向增加 saving 上界的方向保护，不能因 signed complement 展开而丢掉常数或夹掉负 alpha 系数。未完整扫描路径域、未修正负空载项或出现非有限价格时，不得签发新全域证书。

状态：解析有效性已在本文给出；与现有代码非重复已核对。尚未实现、数值验收或实测生产 LB；不宣称对偶退化是唯一停滞原因。
