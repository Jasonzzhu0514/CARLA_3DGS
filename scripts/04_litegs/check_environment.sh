#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/lib/load_config.sh"
PROJECT_DIR="$PROJECT_ROOT"

EXPECTED_COMMIT="ef6597f2155ddd9d5fd02eb2af9c0fe70887faf4"
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

[[ -x "$LITEGS_PYTHON_BIN" ]] || { echo "[失败] Python 不存在：$LITEGS_PYTHON_BIN" >&2; exit 1; }
[[ -d "$LITEGS_DIR/.git" ]] || { echo "[失败] LiteGS 仓库不存在：$LITEGS_DIR" >&2; exit 1; }

actual_commit="$(git -C "$LITEGS_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "[失败] LiteGS 提交不一致：$actual_commit" >&2
  echo "        预期：$EXPECTED_COMMIT" >&2
  exit 1
fi

for file in cameras.bin images.bin points3D.bin; do
  [[ -f "$DATASET/sparse/0/$file" ]] || { echo "[失败] 缺少 $DATASET/sparse/0/$file" >&2; exit 1; }
done
[[ -d "$DATASET/images" ]] || { echo "[失败] 缺少图像目录：$DATASET/images" >&2; exit 1; }

image_count="$(find -L "$DATASET/images" -maxdepth 1 -type f -iname '*.png' | wc -l)"
[[ "$image_count" -gt 0 ]] || { echo '[失败] 图像目录为空' >&2; exit 1; }

export CUDA_HOME="$LITEGS_CUDA_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$LITEGS_PYTHON_BIN" - <<'PY'
import numpy
import torch
import torchmetrics
from simple_knn._C import distCUDA2
import fused_ssim
import litegs_fused
import sys
from pathlib import Path

assert not (Path(sys.prefix) / 'conda-meta').exists(), '预期 uv 虚拟环境'

assert torch.cuda.is_available(), "torch.cuda.is_available() 为 False"
x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
assert (x @ x.T).sum().item() == 83.0
points = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    device="cuda",
)
assert distCUDA2(points).is_cuda
a = torch.rand(1, 3, 16, 16, device="cuda", requires_grad=True)
b = torch.rand(1, 3, 16, 16, device="cuda")
loss = fused_ssim.fused_ssim(a, b).mean()
loss.backward()
assert a.grad is not None
assert torch.isfinite(loss).all() and torch.isfinite(a.grad).all()
quaternion = torch.zeros(4, 128, device='cuda')
quaternion[0] = 1
scale = torch.ones(3, 128, device='cuda')
transform = litegs_fused.createTransformMatrix_forward(quaternion, scale)
assert torch.isfinite(transform).all()
expected = torch.eye(3, device='cuda').unsqueeze(-1).expand_as(transform)
assert torch.allclose(transform, expected)
torch.cuda.synchronize()
print(f"[通过] Python {'.'.join(map(str, __import__('sys').version_info[:3]))}")
print(f"[通过] PyTorch {torch.__version__} / CUDA {torch.version.cuda}")
print(f"[通过] GPU：{torch.cuda.get_device_name(0)}")
print(f"[通过] NumPy {numpy.__version__} / torchmetrics {torchmetrics.__version__}")
print("[通过] simple_knn CUDA、fused_ssim 前后向、litegs_fused CUDA 变换")
PY

echo "[通过] LiteGS 提交：$actual_commit"
echo "[通过] COLMAP 数据：$DATASET"
echo "[通过] PNG 图像数：$image_count"
echo "[完成] 环境检查通过；本脚本没有启动训练。"
