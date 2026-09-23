#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

usage() {
    echo "用法: $0 --dataset <raw session 名称或目录>" >&2
}

INPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--dateset)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            INPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 2
            ;;
    esac
done
[[ -n "$INPUT" ]] || { echo "必须显式指定 --dataset。" >&2; usage; exit 2; }
if [[ -d "${INPUT}" ]]; then
    RAW_DIR="$(cd -- "${INPUT}" && pwd)"
else
    RAW_DIR="${PROJECT_DATA_DIR}/raw/${INPUT}"
fi

if [[ ! -d "${RAW_DIR}/images" || ! -f "${RAW_DIR}/camera_intrinsics.json" ]]; then
    echo "输入 session 不完整: ${RAW_DIR}" >&2
    exit 2
fi
if [[ ! -x "${COLMAP_GPU_PYTHON_BIN}" ]]; then
    echo "找不到 CUDA PyCOLMAP 环境: ${COLMAP_GPU_PYTHON_BIN}" >&2
    echo "请先运行 scripts/03_colmap/setup_gpu_environment.sh" >&2
    exit 2
fi

SESSION_NAME="$(basename -- "${RAW_DIR}")"
WORK_DIR="${PROJECT_DATA_DIR}/colmap/${SESSION_NAME}"
LOG_DIR="${PROJECT_LOG_DIR}/${SESSION_NAME}"
LOG_FILE="${LOG_DIR}/colmap.log"

if [[ -e "${WORK_DIR}" ]]; then
    echo "输出目录已存在，为避免覆盖已停止: ${WORK_DIR}" >&2
    exit 2
fi

mkdir -p "${WORK_DIR}/sparse" "${LOG_DIR}"
ln -s "${RAW_DIR}/images" "${WORK_DIR}/images"
if [[ -d "${RAW_DIR}/masks" ]]; then
    ln -s "${RAW_DIR}/masks" "${WORK_DIR}/masks"
fi
if [[ -d "${RAW_DIR}/images_masked" ]]; then
    ln -s "${RAW_DIR}/images_masked" "${WORK_DIR}/images_masked"
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "COLMAP 预处理开始: ${SESSION_NAME}"
echo "输入图像: ${RAW_DIR}/images"
echo "输出目录: ${WORK_DIR}"
echo "后端: PyCOLMAP CUDA（GPU SIFT 提取与匹配）"

"${COLMAP_GPU_PYTHON_BIN}" "${SCRIPT_DIR}/run_sparse_reconstruction_gpu.py" \
    --raw-dir "${RAW_DIR}" \
    --work-dir "${WORK_DIR}" \
    --gpu-index 0

MODEL_DIR="${WORK_DIR}/sparse/0"
if [[ ! -f "${MODEL_DIR}/cameras.bin" ]]; then
    echo "COLMAP 没有生成 sparse/0 模型。" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/validate_colmap.py" \
    "${WORK_DIR}" \
    --expected-images "$(find "${RAW_DIR}/images" -maxdepth 1 -type f -name '*.png' | wc -l)"

"${COLMAP_GPU_PYTHON_BIN}" "${SCRIPT_DIR}/validate_carla_poses.py" \
    --raw-dir "${RAW_DIR}" \
    --model-dir "${MODEL_DIR}"

echo "COLMAP 预处理完成: ${WORK_DIR}"
