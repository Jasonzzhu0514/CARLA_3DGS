#!/usr/bin/env bash

# Shared machine-local path configuration. Entry points must define PROJECT_ROOT first.
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "PROJECT_ROOT must be set before loading scripts/lib/load_config.sh" >&2
    return 2
fi

PROJECT_CONFIG_FILE="${PROJECT_CONFIG_FILE:-${PROJECT_ROOT}/configs/paths.env}"
if [[ -f "${PROJECT_CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_CONFIG_FILE}"
fi

: "${CARLA_INSTALL_DIR:=${HOME}/Applications/CARLA_0.9.16}"
: "${CARLA_PYTHON_BIN:=${HOME}/.venvs/carla-0.9.16/bin/python}"
: "${PROJECT_DATA_DIR:=${PROJECT_ROOT}/datasets}"
: "${PROJECT_OUTPUT_DIR:=${PROJECT_ROOT}/outputs}"
: "${PROJECT_LOG_DIR:=${PROJECT_ROOT}/logs}"
: "${LITEGS_DIR:=${HOME}/Applications/LiteGS}"
: "${LITEGS_ENV_DIR:=${HOME}/.venvs/litegs}"
: "${LITEGS_PYTHON_BIN:=${LITEGS_ENV_DIR}/bin/python}"
: "${LITEGS_CUDA_DIR:=${CUDA_HOME:-/usr/local/cuda}}"
: "${UV_BIN:=$(command -v uv 2>/dev/null || true)}"
: "${COLMAP_GPU_ENV_DIR:=${HOME}/.venvs/colmap-cuda}"
: "${COLMAP_GPU_PYTHON_BIN:=${COLMAP_GPU_ENV_DIR}/bin/python}"
: "${CAPTURE_RGB_PYTHON_BIN:=python3}"

export PROJECT_DATA_DIR PROJECT_OUTPUT_DIR PROJECT_LOG_DIR
