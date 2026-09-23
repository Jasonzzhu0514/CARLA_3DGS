#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

if [[ ! -x "${CARLA_PYTHON_BIN}" ]]; then
    echo "找不到 CARLA Python 环境: ${CARLA_PYTHON_BIN}" >&2
    exit 1
fi

usage() {
    echo "用法: $0 <session-label> <capture-config.json>" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

SESSION_LABEL="$1"
SESSION_NAME="$(date +%Y%m%d_%H%M%S)_${SESSION_LABEL}"
CAPTURE_CONFIG="$2"
if [[ ! -f "${CAPTURE_CONFIG}" ]]; then
    echo "找不到采集配置: ${CAPTURE_CONFIG}" >&2
    exit 2
fi
SESSION_LOG_DIR="${PROJECT_LOG_DIR}/${SESSION_NAME}"
mkdir -p "${SESSION_LOG_DIR}"

"${CARLA_PYTHON_BIN}" "${SCRIPT_DIR}/capture_surround.py" \
    --config "${CAPTURE_CONFIG}" \
    --output-root "${PROJECT_DATA_DIR}/raw" \
    --session "${SESSION_NAME}" \
    2>&1 | tee "${SESSION_LOG_DIR}/capture.log"

# CARLA 的 save_to_disk 只能写出 RGBA PNG（alpha 恒为 255），而 LiteGS 训练需要
# 3 通道输入，所以采集后必须去掉 alpha。此步骤需要 Pillow。
RGB_PYTHON="${CAPTURE_RGB_PYTHON_BIN:-python3}"
if ! "${RGB_PYTHON}" -c "import PIL" >/dev/null 2>&1; then
    echo "找不到带 Pillow 的解释器: ${RGB_PYTHON}" >&2
    echo "请修改 configs/paths.env 的 CAPTURE_RGB_PYTHON_BIN。" >&2
    exit 1
fi

"${RGB_PYTHON}" "${SCRIPT_DIR}/convert_images_to_rgb.py" \
    "${PROJECT_DATA_DIR}/raw/${SESSION_NAME}" \
    2>&1 | tee -a "${SESSION_LOG_DIR}/capture.log"

if [[ -d "${PROJECT_DATA_DIR}/raw/${SESSION_NAME}/semantic_raw" ]]; then
    "${RGB_PYTHON}" "${SCRIPT_DIR}/build_vehicle_masks.py" \
        "${PROJECT_DATA_DIR}/raw/${SESSION_NAME}" \
        2>&1 | tee -a "${SESSION_LOG_DIR}/capture.log"
fi

"${CARLA_PYTHON_BIN}" "${SCRIPT_DIR}/validate_capture.py" \
    "${PROJECT_DATA_DIR}/raw/${SESSION_NAME}" \
    2>&1 | tee -a "${SESSION_LOG_DIR}/capture.log"
