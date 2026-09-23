#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/lib/load_config.sh"
PROJECT_DIR="$PROJECT_ROOT"

usage() {
  echo "用法: $0 --dataset <COLMAP 数据目录> --model <LiteGS 模型目录>" >&2
}

DATASET=""
MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--dateset)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      DATASET="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MODEL="$2"
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
[[ -n "$DATASET" ]] || { echo "必须显式指定 --dataset。" >&2; usage; exit 2; }
[[ -n "$MODEL" ]] || { echo "必须显式指定 --model。" >&2; usage; exit 2; }
if [[ -d "$DATASET" ]]; then
  DATASET="$(cd -- "$DATASET" && pwd)"
elif [[ -d "$PROJECT_DATA_DIR/colmap/$DATASET" ]]; then
  DATASET="$(cd -- "$PROJECT_DATA_DIR/colmap/$DATASET" && pwd)"
else
  echo "找不到 COLMAP 数据目录：$DATASET" >&2
  exit 2
fi
if [[ -d "$MODEL" ]]; then
  MODEL="$(cd -- "$MODEL" && pwd)"
else
  echo "找不到 LiteGS 模型目录：$MODEL" >&2
  exit 2
fi

export CUDA_HOME="$LITEGS_CUDA_DIR"
export PATH="$LITEGS_ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LITEGS_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$LITEGS_PYTHON_BIN" "$PROJECT_DIR/scripts/04_litegs/evaluate_model.py" \
  --source "$DATASET" \
  --model "$MODEL" \
  --resolution 2 \
  --preview-stride 30
