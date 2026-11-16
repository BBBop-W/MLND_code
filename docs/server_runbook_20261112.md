# Windows 原生服务器交付与验收（2026-11-12）

本交付采用 **Windows x64 + MSVC + CPLEX Concert + CMake**。源码包可在其他平台解包校验，但本轮没有 Windows 主机，也没有执行 Windows 编译或求解。本机 Make 的编译/测试记录见本轮[项目审查](project_assessment_20261112.md)及其 validation JSON；本机未安装 CMake，不能将 Make 通过写成 CMake 或 Windows 已通过。

## 1. 交付边界

`scripts/package_server.py` 只使用 Python 标准库，按目录、扩展名和精确文件名白名单复制当前工作树，不依赖 Git 是否提交。包含 C++ 源码/头文件/探针/构建文件，Python 源码与测试，配置、运行脚本、当前文档，以及两份冻结的派生算例：

- `examples/tsl_2l20h50od/`：C++ 使用的 CSV/JSON 输入；K 由命令参数选择。
- `data/benchmarks/tsl_2l20h50od/`：Python 冻结 NPZ 输入及来源记录，不是原始逐单数据。

不会复制 `outputs/`、历史运行、二进制、Git、CPLEX/Gurobi 安装与许可证、原始订单/地图数据或旧临时目录，也不删除源仓库文件。文档和输入中的历史路径仅为来源记录；部分历史文档链接指向未交付材料。服务器直接重放冻结输入，不能依赖未交付原始数据重新生成它们。

```sh
# 在源码根目录执行；目标必须位于源码目录之外，且尚不存在。
python scripts/package_server.py create --out /tmp/mlnd-server-20261112 --zip
python scripts/package_server.py verify /tmp/mlnd-server-20261112.zip
```

程序生成新目录及同名 `.zip`，逐文件记录 `SERVER_MANIFEST.json` 中的 SHA-256 和字节数，并打印清单与 ZIP 的 SHA-256。已有目录或 ZIP 一律拒绝覆盖。最终打包应在本轮文档、validation JSON 和所有代码完成后进行；不能将进行中的试打包当作最终交付。此清单用于内容一致性检查，不是数字签名，也不证明算法正确或 Windows 兼容。

## 2. Windows 依赖与目录

建议使用短的英文目录，例如 `C:\mlnd\source`、`C:\mlnd\runs`，避免旧 Windows 长路径和窄字符命令行问题。准备：

| 依赖 | 要求与用途 |
|---|---|
| Visual Studio 2022 / Build Tools | 安装“使用 C++ 的桌面开发”、x64 MSVC 工具链、Windows SDK；具体工具链须由服务器的 CPLEX 版本支持。 |
| CMake | 本项目最低 3.18；下例 VS 2022 generator 需要至少 3.21，建议使用已安装且仍受支持的新版。 |
| CPLEX Optimization Studio | 完整 C++/Concert 头文件和 x64 库，服务器自备有效授权；仅安装 Python `cplex` 包不足以构建本程序。 |
| nlohmann/json | `NLOHMANN_JSON_INCLUDE_DIR` 指向包含 `nlohmann/json.hpp` 的目录；不随本源码包附带。本机头文件版本为 3.11.2，服务器应记录自己安装的版本。 |
| Python | 3.11 或更高；用于包校验、独立物理复核与批量调用，不是 C++ 求解器的运行时依赖。 |

先核对实际 CPLEX 的 `cplex\readmeWindows.html`、示例项目和 `c_cpp.html`。IBM 的 22.1.2 安装说明列出 `x64_windows_msvc14` 及 VS 2019/2022 兼容性；这不自动覆盖其他版本。[IBM 安装说明](https://www.ibm.com/docs/en/icos/22.1.2?topic=documentation-installing-cplex-optimization-studio)

本 CMake 的 Windows 配置使用 `cplex/lib/x64_windows_msvc14/stat_mda` 和对应 `concert/lib/.../stat_mda`，配套 `/MD` 与 `_ITERATOR_DEBUG_LEVEL=0`。所有配置共用该发行版运行库；不要混入 `/MT`、`/MDd` 或其他 ABI 的库。`cplex2211.lib`、`cplex2212.lib` 等版本后缀从实际目录发现；存在多个候选时必须显式指定 `-Dcplex_LIBRARY=完整路径`。这些库路径可以通过 `CPLEX_PLATFORM` 和三个 `*_LIBRARY` CMake cache 变量明确覆盖，不能混用安装版本。[IBM 运行库说明](https://community.ibm.com/community/user/discussion/ilocplexpresolvecallbackipresolvecallbacki-was-declared-deprecated)、[CMake `/MD` 配置](https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html)

Windows 分支使用系统 BCrypt 计算 SHA-256，不要求 OpenSSL，也不链接 POSIX 的 `m/pthread/dl`。进度文件使用 `gmtime_s` 和同目录 `MoveFileExW` 替换，覆盖已有进度文件时不先删除旧文件。MSVC 使用 `/fp:strict` 禁止浮点收缩并保留特殊值语义；这不保证与 macOS/Linux 逐位一致，尤其 MSVC 的 `long double` 精度仍需数值验收。[BCrypt](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptcreatehash)、[文件替换](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)、[MSVC 浮点选项](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior)

## 3. 解包、安装与编译

以下为 **PowerShell**，示例安装路径必须替换为服务器真实路径。先在一个全新目录解压 ZIP，再进入源码根目录；不要覆盖旧实验目录。

```powershell
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

Set-Location C:\mlnd\source
py -3.11 scripts/package_server.py verify .
if ($LASTEXITCODE -ne 0) { throw "Source package verification failed" }

py -3.11 -m venv .venv
$Py = (Resolve-Path .venv\Scripts\python.exe).Path
& $Py -m pip install ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }
& $Py -m pip freeze > ..\python-environment-20261112.txt

$CplexRoot = "C:\Program Files\IBM\ILOG\CPLEX_Studio2212"
$JsonInclude = "C:\deps\json\include"
cmake -S cpp -B build-win -G "Visual Studio 17 2022" -A x64 `
  "-DCPLEX_ROOT=$CplexRoot" "-DNLOHMANN_JSON_INCLUDE_DIR=$JsonInclude"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed; inspect the actual installed CPLEX paths" }

# The single list maintained by check_server.py determines all acceptance probes.
$Targets = @("mlnd_cpp", "mlnd_input_probe", "mlnd_path_catalogue")
$Targets += & $Py -c "from scripts.check_server import PROBES; print('\n'.join('mlnd_' + p + '_probe' for p in PROBES))"
if ($LASTEXITCODE -ne 0) { throw "Cannot resolve acceptance targets" }
cmake --build build-win --config Release --target $Targets --parallel 1
if ($LASTEXITCODE -ne 0) { throw "Native build failed" }
```

Python 核心安装不包含 `gurobipy`；上述独立校验和 CPLEX 路线不要求 Gurobi 授权。只有需要历史 Python 优化入口时另装 `".[gurobi]"`。不要把全量 Python 测试都视作无许可证测试，其中仍有 Gurobi 求解用例。

Windows CPLEX callable `.lib` 可能需要安装目录中的对应版本 DLL。库目录与 DLL 目录名称不同；常见 DLL 目录为 `cplex\bin\x64_win64`，以实际安装为准。先列出文件，再将**所选 .lib 对应版本**的 DLL 所在目录加入当前进程 `PATH`，不要从别的机器随意复制 DLL：

```powershell
Get-ChildItem "$CplexRoot\cplex\bin" -Recurse -Filter "cplex*.dll"
$CplexBin = "$CplexRoot\cplex\bin\x64_win64" # 核对上一步输出后设置
if (-not (Test-Path $CplexBin)) { throw "Set CplexBin to the actual installed DLL directory" }
$env:PATH = "$CplexBin;$env:PATH"
```

若更换 CPLEX 版本或 CPU 架构，应重新创建一个构建目录，避免复用旧 CMake cache 的绝对库路径。Makefile 保留 macOS/Linux 使用方式，不是 Windows 原生入口。

## 4. 服务器验收

先运行读取和纯 Python 工具测试，再执行小规模求解验收。以下输出目录也必须全新。

```powershell
& .\build-win\Release\mlnd_cpp.exe --input examples\tsl_2l20h50od --k 2 --inspect
if ($LASTEXITCODE -ne 0) { throw "Native input/hash validation failed" }
& $Py -m unittest discover -s tests\unit -p test_server_package.py -v
if ($LASTEXITCODE -ne 0) { throw "Packaging tests failed" }
& $Py -m pytest tests\unit\test_server_experiment.py tests\unit\test_optional_solver.py -q
if ($LASTEXITCODE -ne 0) { throw "Python entrypoint tests failed" }
& $Py scripts\check_server.py --binary-dir build-win\Release --out C:\mlnd\checks\acceptance-20261112
if ($LASTEXITCODE -ne 0) { throw "Server acceptance failed; inspect acceptance.json and per-probe logs" }
& $Py scripts\package_server.py verify . --allow-extra-files
if ($LASTEXITCODE -ne 0) { throw "Packaged source changed during installation or tests" }
```

`check_server.py` 顺序运行 21 组探针，再调用 `cpp/tests/task_migration_reference.py`：后者在 L2/H2/OD2 小例上比较 CSV/1 与 CSV/2、K1/K2、两种算法，共八次限时求解，并由 Python 重算物理可行性。Windows 自动使用 `.exe`，路径/输入/模型比较与求解退出状态分开记录；结果位于新目录 `acceptance.json`。通过表明这些构造用例满足检查，不能替代大规模性能证据或全部浮点平台的形式证明。

安装和编译后包目录新增 `.venv/`、构建产物等，严格包校验会报告额外文件。此时 `--allow-extra-files` 只复核清单列出的原始交付文件，不扫描新增输出；不是忽略已列源码的修改。

## 5. 冻结算例运行

使用跨平台入口 `scripts/run_server_experiment.py`，每次给新的父目录；它为原生求解和 Python 独立复核分别记录状态，并保存源码、输入和二进制 SHA-256。`--seconds` 是原生总预算，外层进程默认墙钟上限为该预算加 120 秒，可用 `--wall-limit` 明确设置；物理复核另有默认 300 秒上限。

```powershell
$Binary = (Resolve-Path build-win\Release\mlnd_cpp.exe).Path
$Instance = (Resolve-Path examples\tsl_2l20h50od).Path
foreach ($K in 1, 2) {
  foreach ($Preset in "performance-20261105", "full-cplex-20261029") {
    $Run = "C:\mlnd\runs\20261112-K$K-$Preset"
    & $Py scripts\run_server_experiment.py --binary $Binary --instance $Instance `
      --out $Run --preset $Preset --k $K --seconds 600 --gap .01 --threads 1 --seed 43
    if ($LASTEXITCODE -ne 0) { throw "Run or validation failed: $Run" }
  }
}
```

运行器在 K1 或非 L2 时自动关闭四个限定于 L2/K2 的 ALNS 相关开关，记录实际关闭项；仍应核对 `resolved_configuration`。此示例串行运行，不能在其他未记录并发负载下把墙钟差直接解释为算法差异。保留每次父目录的 `experiment.json` 与原生 `run/`；分别报告独立验证后的 UB、认证 LB、gap、终止原因、目标 gap 是否达到、原生与进程时间。达到时间上限或外层被终止不代表证明最优；缺少有效最终产物的运行也不能仅据 checkpoint 当作成功运行。

## 6. 本轮修复与尚待确认

已修复的确定平台阻断包括：Windows 被当作 Linux 的库路径与编译参数；无版本后缀的 `cplex` 库查找；Windows 不必要的 OpenSSL/POSIX 链接；`gmtime_r`；验收脚本的 `.exe` 定位。进度输出改为显式 Win32 替换并增加重复写入、UTC 格式和替换失败保留原目标的回归。MSVC 的 `std::filesystem::rename` 本身也支持替换已有普通文件，因此这项整理不是“原实现第二次写入必失败”的修复。[Microsoft STL 实现](https://github.com/microsoft/STL/blob/main/stl/src/filesystem.cpp)

服务器验收前仍需确定 CPLEX 版本、MSVC 工具链、JSON 头版本、Windows 版本和实际安装路径。当前文档依据本地源码、已获取的官方 API 文档及 IBM 安装说明搜索正文；IBM 部分完整网页随后访问失败。相关官方链接和服务器随安装提供的示例是核对入口，不能据此宣称已经在目标服务器运行。
