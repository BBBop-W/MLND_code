# 南山全量整数 Benders 实验

入口：[实验简报](report.html)；[完整方法](METHOD.html)。

本目录的正式实验使用整数设施／OD 基地对主问题。可用率、idle 和所有运营流量均在内层。历史部分 Benders 实验仍在原目录保存；本目录的驱动程序使用 `nanshan_integer.py`。

## 复现

从项目根目录运行，需现有 Gurobi 许可及 NumPy、Numba、SciPy、Matplotlib：

```bash
python -m mlnd_path.nanshan_timing
python -m mlnd_path.benchmark_nanshan
python -m mlnd_path.benchmark_nanshan --levels 120 --seconds 300 --output mlnd_path/nanshan_integer_20260914/extended_300s
python -m mlnd_path.validate_nanshan mlnd_path/nanshan_integer_20260914/formal_60s mlnd_path/nanshan_integer_20260914/extended_300s --output mlnd_path/nanshan_integer_20260914/validation.json
python -m mlnd_path.render_nanshan_report
```

六档完整实例为 H11、H24、H43、H68、H87、H120。每档依次运行基础版与主问题加强版；不同档交替两种方法的运行次序。所有优化串行、单线程、固定种子 43，目标 gap 为 0.1%。运行同时记录设定预算与实际墙钟时间。

## 主要代码

| 文件 | 内容 |
|---|---|
| `nanshan_oracle.py` | 历史数据适配，携带剩余电量的隐式完整任务定价 |
| `nanshan_timing.py` | 与需求窗口一致的取餐前置时间经验分布 |
| `integer_recourse.py` | 固定整数设计的小型 SOCP、显式锥对偶及全候选补全 |
| `integer_master_cuts.py` | 共享 idle 的子模链割与一维闭式下界 |
| `nanshan_integer.py` | 整数主问题、成本覆盖割、回调与初始可行方案 |
| `benchmark_nanshan.py` | 串行对照与逐运行记录 |
| `validate_nanshan.py` | 保存方案的独立成本、机队、流平衡及源文件核验 |

相关测试入口：

```bash
python -m pytest -q tests/test_mlnd_path.py tests/test_mlnd_path_integer.py tests/test_mlnd_path_integer_recourse.py tests/test_mlnd_path_nanshan_oracle.py
```

## 输出结构

- `formal_60s/`：六档、两方法的正式对照。
- `extended_300s/`：最大档、两方法的延长实验。
- `timing/`：排序经验样本、前缀和、来源哈希及过滤统计；不改变原始 OD 需求。
- `smoke/`、`pilot/`：开发试跑，包括早期确定性前置时间版本；不混入正式汇总。
- `validation.json`、`summary.csv`：验算及逐运行汇总。
- `data_audit.md`：历史数据、成本参数与模型口径逐项来源。

每次正式运行保存源文件哈希、原始数据哈希、预算、实际时间、上下界、可行设计、可用率、idle 和空返矩阵。求解时间包含路径预处理、主问题建立、初始方案和全部迭代；数据读取与一次 JIT 预热单独记录。

原论文 `MLND/main.tex` 的既有工作区修改保持原样；本轮校验其 SHA-256 为 `38db84c6a7bfb40c2b22b526c5b74981be551f11da575fb483dd090a4b64dcad`。
