#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

if [[ ! -x "${UV_BIN}" ]]; then
    echo "找不到 uv: ${UV_BIN}" >&2
    exit 1
fi
if [[ ! -x "${COLMAP_GPU_ENV_DIR}/bin/python" ]]; then
    "${UV_BIN}" venv --python 3.10 --managed-python "${COLMAP_GPU_ENV_DIR}"
fi
"${UV_BIN}" pip install --python "${COLMAP_GPU_PYTHON_BIN}" \
    pycolmap-cuda12==4.2.0

"${COLMAP_GPU_PYTHON_BIN}" - <<'PY'
import pycolmap
assert pycolmap.has_cuda, "安装的 PyCOLMAP 没有 CUDA 支持"
print(f"[通过] PyCOLMAP {pycolmap.__version__}, CUDA={pycolmap.has_cuda}")
PY
