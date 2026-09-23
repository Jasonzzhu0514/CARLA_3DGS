#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/lib/load_config.sh"

if [[ ! -x "${CARLA_PYTHON_BIN}" ]]; then
    echo "找不到 CARLA Python 环境: ${CARLA_PYTHON_BIN}" >&2
    exit 1
fi

exec "${CARLA_PYTHON_BIN}" "${SCRIPT_DIR}/spawn_vehicle.py" "$@"
