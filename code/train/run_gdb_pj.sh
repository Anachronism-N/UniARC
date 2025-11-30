#!/usr/bin/env bash
set -euo pipefail

# 说明：在 pj conda 环境中启动 gdb，加载 train_llama_continus.gdb 调试脚本。
# 使用：
#   bash ASRCompare/code/train/run_gdb_pj.sh [脚本参数...]
# 例子：
#   bash ASRCompare/code/train/run_gdb_pj.sh --epochs 1 --batch-size 4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY_SCRIPT="ASRCompare/code/train/train_llama_continus.py"

# 优先使用系统 conda，其次使用本地 miniconda3
if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  CONDA_CMD="$REPO_ROOT/miniconda3/bin/conda"
fi

cd "$REPO_ROOT"

# 构造 gdb 命令，附带 set args 以传递 Python 脚本参数
GDB_CMD="gdb -q -x ASRCompare/code/train/train_llama_continus.gdb -ex \"set args $PY_SCRIPT $*\""

# 优先使用 conda run 进入 pj 环境运行 gdb，失败则回退为手动激活
if "$CONDA_CMD" run -n pj gdb --version >/dev/null 2>&1; then
  "$CONDA_CMD" run -n pj bash -lc "$GDB_CMD"
else
  # 回退方案：source 激活 pj 环境后执行 gdb
  # shellcheck disable=SC1091
  source "$REPO_ROOT/miniconda3/bin/activate" pj
  eval "$GDB_CMD"
fi