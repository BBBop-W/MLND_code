# 当前代码职责与验证入口

以任务级独立出发基地指派、固定广义任务需求和骑手送到家时限为当前数值模型。`consecutive` 名称和日期 preset 本身不能识别历史模型，结果还必须检查 `assignment_scope="task"`、`delivery_time_scope="courier_home"` 及源码／输入指纹。

## 生产求解与独立校验

```text
标准 CSV/JSON + manifest
  → cpp/main → input → paths + preprocessed_demand
  → formulation → 完整 CPLEX，或 benders + pricing + 上界搜索
  → 原生 result / solution / path_flows
  → Python native.validate
  → 从完整输入域重建任务率，再独立重算保存点的物理约束与成本
```

完整模型与 Benders 共用路径资源和模型矩阵。完整目录定义定价认证的范围；RMP 只存部分列。受限列上界搜索、ALNS、稳定化候选主问题产生候选点，不能贡献原问题全局下界。全局下界来自具有完整域证书的割和全域 master。

| 职责 | 代码 | 不变量 |
|---|---|---|
| 配置、预算、原生入口 | `cpp/src/main.cpp`、`solver_presets.cpp` | 记录最终生效配置；拒绝不相容选项 |
| 输入与固定任务重构 | `input.cpp`、`preprocessed_demand.cpp` | 原始 OD 守恒，AA 双计数；任务率在设计前固定 |
| 物理路径 | `paths.cpp`、`include/mlnd/path_identity.hpp` | 返回基地与物理资源属于列身份；home 时限不换为混合时限 |
| 公共矩阵与求解接口 | `formulation.cpp`、`linear_model.cpp` | 完整模型、固定设计和列系数同源 |
| Benders、定价与证书 | `benders.cpp`、`pricing.cpp`、`dual_workspace.cpp` | 完整域定价；单设计证书不直接作为全局 LB |
| 结构割 | `task_savings.cpp`、`airline_resources.cpp`、`queue_polymatroid.cpp`、`design_face_cuts.cpp` 等 | 对应域和状态验证完成才发割 |
| 上界搜索 | `heuristics.cpp`、`primal_alns.cpp`、`primal_support.cpp` | 保留 incumbent 支持；受限模型局部界不外传 |
| 保存与进度 | `output.cpp`、`progress.cpp` | 稳定 ID；完整替换 checkpoint；清晰终止状态 |
| 物理校验 | `src/experiments/native.py`、`src/operations/validation.py` | 重新检查输入哈希、任务率、路径、资源、订单、成本与上下界 |
| 共享 Python 路径遍历 | `src/model/paths.py`、`generalized_demand.py` | 同一生成器提供完整收集与流式统计；中断不提供完整证据 |

独立校验不调用优化器，也不信任结果中的路径系数、汇总值或删减后的任务目录。当前 preprocessed 校验必须完整扫描输入物理域来验证固定任务率；该步骤独立计时，流式统计不保留全部服务列。保存点的资源与约束仍逐项检查。

## 服务器入口的收拢

| 入口 | 用途 |
|---|---|
| `scripts/package_server.py` | 白名单源码包与逐文件校验，排除旧实验、机器二进制和许可证 |
| `cpp/CMakeLists.txt` | Windows/MSVC 与 POSIX 平台各自构建；Windows 必须本机编译 |
| `scripts/check_server.py` | 串行小型验收和 CSV/1、CSV/2、K1、K2 端到端比较 |
| `scripts/run_server_experiment.py` | 单例运行、外部时间限制、源码／输入／二进制指纹及独立校验 |

服务器安装与命令见 [runbook](server_runbook_20261112.md)。本地 macOS 验证不能替代 Windows 实机编译、许可证探针和数值验收。

## 保留的历史代码

`src/solvers` 是历史 Python/Gurobi 模型，不是当前固定任务 C++ 模型的对照优化器；优化依赖通过 `.[gurobi]` 按需安装。`src/experiments` 中日期命名脚本、BDD 诊断和历史模型文档保留其原始复现用途。当前服务器入口不调用其中依赖 POSIX `wait4`、进程组和 `fcntl` 的历史批量调度器。

本轮不拆分 `benders.cpp` 的状态机，也不删除历史参数组合。先固定服务器可复现基线与消融证据，再按测试覆盖提取内部职责，避免一次重排同时影响数学证书、调度和搜索轨迹。
