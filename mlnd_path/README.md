# MLND_Path：路径模型与算法规模测试

这是 `MLND_Path` 分支中的独立实验实现。模型以基地为闭排队网络库存节点，完整任务列包含取餐柜、目的柜、跨阶段电量和中途换电。整数层选择设施与 OD 基地对；连续层决定可用率、服务列流量和空重定位。正的机队持有成本使连续机队变量可以消去。

结果阅读入口为 [算法测试简报](BENCHMARK_REPORT.md)，可视版为 [benchmark_report.html](benchmark_report.html)。正式实验的参数、源文件哈希、逐实例解和轨迹保存在 [benchmarks/multiscale_20260914](benchmarks/multiscale_20260914)。

## 运行

需要 Python 3.11+、Gurobi 及有效许可证、NumPy、SciPy。当前环境验证使用 Python 3.13.9、Gurobi 13.0.2、单线程。

```bash
python -m pytest tests/test_mlnd_path.py -q
python -m mlnd_path.verify_instances
python -m mlnd_path.benchmark \
  --scales tiny small medium large --seeds 11 29 \
  --algorithms full full_strong benders_lazy benders_root \
  --time-limit 30 --mip-gap 0.0001 \
  --output mlnd_path/benchmarks/my_run
```

`tiny/small/medium/large/stress` 的基地／柜／OD／远程换电站数量依次为 `2/3/2/1`、`2/4/4/1`、`3/5/8/1`、`3/6/12/2`、`4/8/24/2`。也可调用 `make_instance(n_l,n_h,n_od,n_k,seed)` 创建其他规模。增加 `--save-instances` 可保存完整列数据；默认通过固定种子、参数和代码哈希复现。

## 当前模型口径

- 订单为固定强度的 Poisson 请求，缺机订单采用全程骑手兜底。
- 每个 OD 整数选择至多一对基地，柜对与换电路线按连续流量分配。
- 随机取餐历时采用给定均值的指数分布，使用解析的 `E[max(A, ξ+g)]`；优化没有样本场景。
- 给定目的柜的自取／到家比例由 Logit 计算，折入列成本。
- 电量沿完整任务连续传递；H 不恢复电量，实际换电才恢复满电，返回基地后执行终端换电。
- 换电服务即时、库存充足；当前机队按正单位持有成本内生规划，无硬机队上限。
- 流量守恒、有限空重定位时钟和普通 idle 二阶锥均按完整中文模型实现。

全部规模测试采用合成几何、需求和费用，实例元数据给出具体取值与单位。

## 完整候选列的精确预处理

对每个商家、基地对和柜对，枚举允许使用的换电设施子集，使用资源标签法求该子集下费用最低的完整路径，再对实际路线去重。这里的费用为运营费加 `fleet_cost × expected_duration`，跨阶段持续扣减电量；路线没有人为跳数或 top-K 截断。空程采用相同方法。

**整数设计值函数等价。** 固定任一整数设施设计，其实际开放的换电设施构成某个子集。对每个 OD、基地对和柜对，候选并集中包含该子集下费用最低的可行路径。用它替换同类任一路径，保持 OD 服务量、基地流量平衡、柜激活及 idle 变量，并降低或保持目标。换电激活行采用安全的大 M，设施开放时容纳全部流量。因此，当前模型每个整数设计的最优运营值 `Q(v)` 被精确保留。空程替换同理。

定价扫描这个固定的、保留整数设计值函数的完整列集，包含使用当前关闭设施的候选列。预处理列并集的分数设施松弛可以与全物理路径模型不同。增加硬机队上限或实际换电容量时，预处理需要同时保留时间、费用和资源使用的权衡。

独立穷举验证见 `verify_instances.py`：比较允许设施子集中的最优路径与资源标签、列并集得到的结果，同时重算跨阶段电量。

## 求解方法

| 名称 | 内容 |
|---|---|
| `full` | 完整候选列 MISOCP，Gurobi 直接求解 |
| `full_strong` | 增加整数有效的流量与设施激活加强约束后直接求解 |
| `benders_loop` | 反复求整数主问题，连续 SOCP 内用 CG，回传完整定价后的割 |
| `benders_lazy` | 在一次分支搜索中使用 lazy Benders 割，复用列池 |
| `benders_strong` | 加强约束与 lazy Benders＋CG |
| `benders_root` | 加强约束、根节点分数设计割、lazy Benders＋CG |
| `benders_pareto` | 在完整定价后进一步选择改善核心点下界的对偶割 |

`benders_root` 将至多 25% 的剩余预算、上限 8 秒用于根节点分离，最多 80 轮，然后继续整数分支。分数设计的运营方案只用于生成下界割，整数可行上界来自实际整数设计的运营解。

### 加强约束

令 `T_od,l` 为 OD 从基地 l 出发的实际总任务率，`B_od,l` 为其来源选择变量之和，`M_od=h_od*rho_max`。加入

```text
T_od,l <= h_od * rho_l
h_od * rho_l - T_od,l <= M_od * (1 - B_od,l)
```

结合已有 `T_od,l <= M_od * B_od,l`，得到整数来源与连续可用率关系的线性凸包。柜和换电点激活另外按 OD 拆分；空程换电激活按起点拆分。全部基准行保留，所有整数设计下的原运营可行解保持不变。

### 下界与数值证书

连续子问题独立建立 SOC 对偶，完整扫描候选列。生成割时，对线性约束乘子修正符号，并在拉格朗日函数内解析最小化 idle 项：

```text
min { cN*I + a*rho : I >= rho/(1-rho), rho >= 0 }
= 0                                      if a >= -cN
= -(sqrt(-a) - sqrt(cN))^2                otherwise.
```

再按各 OD 的总服务上界、各基地的空程上界修正微小负检验数。只有完整定价和原始／修正对偶间隙通过检查后，外层才采用该割。主问题的全局 bound 更新下界，已验证的运营原始解更新上界；超时保留两者。

Pareto 选割参照 [Magnanti and Wong (1981)](https://pubsonline.informs.org/doi/abs/10.1287/opre.29.3.464)：约束生成点的对偶目标接近最优，改善内部核心点的对偶值，并重新进行完整候选定价。程序按最终校正后的改善决定是否采用新割。

## 模块

| 文件 | 功能 |
|---|---|
| `data.py` | 实例与全局列编号合同 |
| `instances.py` | 合成几何、解析期望、utility、精确电量路径标签 |
| `solver.py` | 完整 MISOCP、连续 SOCP、显式对偶与 CG |
| `benders.py` | 整数主问题、循环／lazy 分解、上下界与时间预算 |
| `strengthening.py` | 保留整数可行解的加强约束 |
| `root_cuts.py` / `pareto.py` | 根节点分离／可选对偶选割 |
| `solution_metrics.py` | 重算服务方式、实际换电、机队与流量平衡 |
| `benchmark.py` | 顺序单线程、多规模、多种子实验与原始记录 |

当前 CG 用于减少连续求解器中的活动列，预处理列全集仍保存在内存中。下一步的实现重点可以根据简报中的主问题、建模、连续求解和定价占比确定。
