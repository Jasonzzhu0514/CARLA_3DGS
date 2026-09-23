#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

if [[ ! -x "${CARLA_INSTALL_DIR}/CarlaUE4.sh" ]]; then
    echo "找不到可执行文件: ${CARLA_INSTALL_DIR}/CarlaUE4.sh" >&2
    exit 1
fi

cd "${CARLA_INSTALL_DIR}"
exec ./CarlaUE4.sh \
    -quality-level=Low \
    -windowed \
    -ResX=800 \
    -ResY=600
