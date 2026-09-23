#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/lib/load_config.sh"
PROJECT_DIR="$PROJECT_ROOT"

usage() {
  echo "用法: $0 --dataset <COLMAP 数据目录>" >&2
}

DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--dateset)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      DATASET="$2"
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
if [[ -d "$DATASET" ]]; then
  DATASET="$(cd -- "$DATASET" && pwd)"
elif [[ -d "$PROJECT_DATA_DIR/colmap/$DATASET" ]]; then
  DATASET="$(cd -- "$PROJECT_DATA_DIR/colmap/$DATASET" && pwd)"
else
  echo "找不到 COLMAP 数据目录：$DATASET" >&2
  exit 2
fi
SESSION="$(basename "$DATASET")"
RUN_TIME="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$PROJECT_OUTPUT_DIR/$SESSION/litegs_smoke_$RUN_TIME"
LOG_DIR="$PROJECT_LOG_DIR/$SESSION"
LOG_FILE="$LOG_DIR/litegs_smoke_$RUN_TIME.log"

"$PROJECT_DIR/scripts/04_litegs/check_environment.sh" --dataset "$DATASET"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export CUDA_HOME="$LITEGS_CUDA_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "[训练] 小规模可行性检查：720 次图像迭代，分辨率缩小 2 倍"
echo "[输出] $OUTPUT_DIR"
echo "[日志] $LOG_FILE"

cd "$LITEGS_DIR"
"$LITEGS_PYTHON_BIN" example_train.py \
  --sh_degree 3 \
  -s "$DATASET" \
  -i images \
  -m "$OUTPUT_DIR" \
  --resolution 2 \
  --iterations 720 \
  --densify_until 0 \
  2>&1 | tee "$LOG_FILE"

PLY="$OUTPUT_DIR/point_cloud/finish/point_cloud.ply"
[[ -s "$PLY" ]] || { echo "[失败] 未生成最终 PLY：$PLY" >&2; exit 1; }
echo "[完成] 小规模训练已退出并生成 PLY，请结合日志与模型检查质量：$PLY"
