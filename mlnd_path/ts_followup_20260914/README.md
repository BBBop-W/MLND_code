# TS 文献对照与第二轮算法实验

主报告：[可直接阅读的 HTML](report.html) · [Markdown](TS_FOLLOWUP_REPORT.md)。

## 方法与证据

- `literature_audit.md`：TS 原文逐项核对，含期刊页码、公式与实际场景数。
- `partial_algorithm.md`：保留连续运营骨架的部分 Benders，路径 LP 的两阶段可行性及成本割。
- `analytic_path_projection.md`：固定整数设施后的逐组解析路径值，全候选成本割与可达性割证明。
- `validation_tests.xml`：32 项独立测试的最终记录。

## 正式实验

所有运行均单线程、30 秒求解预算、目标 gap 0.01%，使用相同合成实例和物理模型。路径预处理另外计时，原始结果保留端到端用时。七种方法均保持连续可用率。

| 目录 | 运行内容 | 次数 |
|---|---|---:|
| `benchmarks_core` | 中型／较大型各两个种子；完整 MISOCP、原锥 Benders、部分 Benders 全列 LP／CG | 16 |
| `benchmarks_analytic` | 相同四个实例；解析路径组多割 | 4 |
| `benchmarks_stress` | 81,744 服务列实例；上述五种方法 | 5 |
| `benchmarks_oa` | 四个中型／较大型实例；idle 外逼近 | 4 |
| `benchmarks_oa_inject` | 四个中型／较大型实例；外逼近并提交可行解 | 4 |
| `benchmarks_oa_stress` | 扩展实例；idle 外逼近 | 1 |
| `benchmarks_oa_stress_inject` | 扩展实例；外逼近并提交可行解 | 1 |

合计 5 个实例、35 次正式运行。每个目录均有配置、源码哈希、汇总、实例元数据及逐次完整结果。结果 JSON 采用紧凑存储，完整列编号下的流量数组和迭代轨迹均予保留。

`pilot_partial`、`pilot_oa`、`pilot_oa_inject` 是开发预实验，预算和代码阶段与正式实验有区别，未并入报告数据。`validation_partial.xml` 是 OA 实现之前的阶段性验证。

## 复算

```bash
python -m mlnd_path.benchmark_followup --scales medium large --seeds 11 29 --time-limit 30 --output /tmp/mlnd_core_rerun
python -m mlnd_path.ts_followup_20260914.run_oa --scales medium large --seeds 11 29 --time-limit 30 --inject --output /tmp/mlnd_oa_rerun
python -m pytest tests/test_mlnd_path.py tests/test_mlnd_path_partial.py tests/test_mlnd_path_analytic.py tests/test_mlnd_path_oa.py -q
```

报告生成器 `build_report.py` 接收正式实验目录列表，读取保存记录、核对计算源码哈希与跨算法上下界，并生成自包含 HTML。改变计算代码后，应使用新目录重新运行实验。
