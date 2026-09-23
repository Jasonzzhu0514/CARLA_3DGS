#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

failed=0

check_file() {
    local description="$1"
    local path="$2"
    if [[ -e "${path}" ]]; then
        echo "[通过] ${description}: ${path}"
    else
        echo "[失败] ${description}: ${path}" >&2
        failed=1
    fi
}

check_file "CARLA 启动脚本" "${CARLA_INSTALL_DIR}/CarlaUE4.sh"
check_file "Python 解释器" "${CARLA_PYTHON_BIN}"

if [[ -x "${CARLA_PYTHON_BIN}" ]]; then
    "${CARLA_PYTHON_BIN}" - <<'PY'
import sys
import carla

print(f"[信息] Python: {sys.version.split()[0]}")
print(f"[信息] CARLA 模块: {carla.__file__}")
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used \
        --format='csv,noheader' | sed 's/^/[信息] GPU: /'
else
    echo "[提示] 未找到 nvidia-smi。"
fi

if ss -ltnH '( sport = :2000 )' | grep -q .; then
    echo "[信息] CARLA 端口 2000 正在监听。"
else
    echo "[信息] CARLA 端口 2000 当前未监听；运行 start_carla.sh 后应开始监听。"
fi

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "[通过] 基础文件和 Python 环境检查完成。"
