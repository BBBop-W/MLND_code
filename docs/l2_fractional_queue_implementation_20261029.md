# 在当前分数主问题处分离 queue 割

本轮承接2026-10-28的价格搜索实验：M08在整数recourse返回后加入123条native价格割，但相对同轮刚加入的ordinary割，其同点增益全部低于0.000084。完整域有效性已经验证；本轮改变分离时机和接受规则，目标是提高新增行相对当前主问题的边际作用。

## 策略和认证边界

新增默认关闭的`--queue-fractional-separation`，要求`--queue-price-search --queue-polymatroid-cuts`。关闭时保留上一轮算法行为。开启后，外层root/integer recourse仅缓存最新合法native资源价格及来源候选，不在刚闭合的整数点进行原有post-CG价格搜索。

在下一次root或integer搜索阶段，用当前master完整的变量界、目标常数/系数、所有行及最新RHS建立副本，仅取消binary标志。它是当前全局主问题的LP松弛；包含canonical必要支持行时，scope为`optimality_preserving_master_relaxation`，不与旧运营连续松弛混用。

只有CPLEX报告optimal、有解、通过原矩阵可行性检查且目标有限时，才以`objective - 1e-7*(1+abs(objective))`形成LP认证下界。真实主问题接纳临时LP使用的全部新行后，才可以保留该下界。任何候选解值、队列割同点值、分离违反量、profile池增益都不能直接提升全局LB。已经认证的旧LB始终保留。

对LP求得的分数设计，用缓存native价格和当前最强资源profile做原7点评价；每个新价格仍完整扫描物理路径域，signed B、负空驶修正、缺失分支AND与EPI构造均沿用原实现。接受条件为原clamp分离点与实际LP点的违反量都严格超过`1e-5*(1+abs(LP objective))`，且行有效、违反、未重复。这里的theta来自包含全部已有ordinary/queue约束的LP解，因此比较对象是当前master，不是旧queue profile池。

每次至多3次LP、2条新行；末轮只重求LP认证。若当前LP点已是整数则不做价格搜索。失败、预算不足、非最优或不可行LP不产生新认证值；已有独立有效行和先前已认证下界仍可保留。保存原clamp点证书，同时记录未clamp LP坐标与违反量供独立复算，避免边界数值误差被掩盖。

## 时间与调度

沿用已确认的公开seam：价格/分离API、原完整物理recourse、全流程求解及冻结终态审计。新增公共控制函数负责时间、fresh donor和pending recourse优先级。

- 每次最多分配2秒，LP与price各最多1秒；每个LP solve最多0.3秒，每次price search最多0.5秒。
- LP构造、求解、验证、同步和记录开销累计额度为总时限2%；price累计额度为2%，包含原完整路径扫描和搜索，两者不重复计时。
- 距上次调用结束至少5秒；仅使用新的latest donor，每个来源序号最多触发一次调用。尚有pending recourse时不插入此工作。
- 原target master refinement、上界搜索与总600秒截止均保持原语义。分配额度和实际墙钟耗时分开保存，末次同步/求解轻微超时按实报告。

## 公共验证与正式比较

测试首先构造独立手算的两基地模型：已闭合整数anchor上queue与ordinary近等价，但当前master LP在分数点仍缺一条排列EPI。检验新增行确实切掉当前LP点，并通过重求LP提高认证目标；这是公共API正确性用例，不作为L2生产性能证据。另检查完整物理recourse跨点有效性、预算/失败边界和32项控制规则，再检查K1/K2集成。

正式比较只有两种算法、三个L2实例，共六次新运行：baseline=`price_target_baseline`，即上一轮native价格搜索与target精化；treatment=`fractional_queue`，只额外启用新分离策略。两方共享新二进制、物理输入、K=2、seed43、单线程、600秒、1%目标和现有上界机制；不引入历史warm start或外部root证书。

分别报告terminal可行UB、认证LB、gap、时间、终止原因，以及当前LP包络被提高的证据、接受/拒绝次数、LP/price耗时。新旧源文件与二进制、输入、协议、审计器均冻结留档。独立审计从上一轮修复后的版本继承：负空驶项相对于独立精确需求检查危险方向，不信任保存的总需求字段代替输入。
