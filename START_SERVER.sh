#!/usr/bin/env bash
# Linux entrypoint for the current Git branch; old package source is never used.
set -Eeuo pipefail
REPO=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_DIR="$REPO/outputs/server_environment"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$REPO/outputs/server_matplotlib_cache"

for argument in "$@"; do
    if [[ "$argument" == --status || "$argument" == --help || "$argument" == -h ]]; then
        PY=$(command -v python3 || command -v python)
        exec "$PY" "$REPO/scripts/server_start.py" "$@"
    fi
done
[[ $(uname -s) == Linux ]] || { echo '请在 Linux 服务器运行 START_SERVER.sh。' >&2; exit 2; }
command -v flock >/dev/null || { echo '缺少 flock（util-linux），请联系服务器管理员。' >&2; exit 2; }
GLOBAL_LOCK=${MLND_LOCK_FILE:-"$(dirname -- "$REPO")/.mlnd-capacity-run.lock"}
exec 9>"$GLOBAL_LOCK"
flock -n 9 || { echo '已有服务器批次或准备任务在运行；请先用 --status 查看。' >&2; exit 75; }
export MLND_LAUNCH_LOCK_FD=9
CONDA_BIN=${CONDA_EXE:-}
if [[ -z "$CONDA_BIN" ]]; then CONDA_BIN=$(command -v conda || true); fi
if [[ -z "$CONDA_BIN" ]]; then
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
    done
fi
[[ -n "$CONDA_BIN" ]] || { echo '未找到 Conda；请在已有 Conda 终端执行。' >&2; exit 2; }
CONDA_BASE=$("$CONDA_BIN" info --base)
mkdir -p "$REPO/outputs"
exec 8>"$REPO/outputs/.server-environment.lock"
flock -n 8 || { echo '另一启动器正在准备环境，请稍后再试。' >&2; exit 75; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$ENV_DIR" --override-channels -c conda-forge \
        python=3.11 cmake ninja cxx-compiler openssl nlohmann_json pip
fi
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
set -u
REQ_HASH=$(cat "$REPO/requirements.txt" "$REPO/requirements-server.txt" | sha256sum | cut -d ' ' -f 1)
if [[ ! -f "$ENV_DIR/.mlnd-requirements.sha256" || $(cat "$ENV_DIR/.mlnd-requirements.sha256") != "$REQ_HASH" ]]; then
    "$ENV_DIR/bin/python" -m pip install -r "$REPO/requirements-server.txt"
    printf '%s\n' "$REQ_HASH" > "$ENV_DIR/.mlnd-requirements.sha256"
fi
exec 8>&-
exec "$ENV_DIR/bin/python" "$REPO/scripts/server_start.py" "$@"
