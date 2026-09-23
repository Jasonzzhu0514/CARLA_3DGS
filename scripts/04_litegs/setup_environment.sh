#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/lib/load_config.sh"
PROJECT_DIR="$PROJECT_ROOT"

LITEGS_REPO_URL="https://github.com/MooreThreads/LiteGS.git"
LITEGS_COMMIT="ef6597f2155ddd9d5fd02eb2af9c0fe70887faf4"

if [[ ! -x "$UV_BIN" ]]; then
  echo "[失败] 请先安装 uv：$UV_BIN" >&2
  exit 1
fi

if [[ ! -d "$LITEGS_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LITEGS_DIR")"
  git clone --recursive "$LITEGS_REPO_URL" "$LITEGS_DIR"
fi

git -C "$LITEGS_DIR" fetch --all --tags
git -C "$LITEGS_DIR" checkout "$LITEGS_COMMIT"
git -C "$LITEGS_DIR" submodule update --init --recursive

if [[ ! -x "$LITEGS_ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv --python 3.10 --managed-python "$LITEGS_ENV_DIR"
fi

PYTHON_BIN="$LITEGS_ENV_DIR/bin/python"
"$UV_BIN" pip install --python "$PYTHON_BIN" \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
"$UV_BIN" pip install --python "$PYTHON_BIN" -r "$PROJECT_DIR/configs/requirements/litegs.txt"

# NVIDIA 官方发布的二进制工具包，直接解压，不调用 Conda。
if [[ ! -f "$LITEGS_CUDA_DIR/include/cub/cub.cuh" || ! -x "$LITEGS_CUDA_DIR/bin/nvcc" ]]; then
  mkdir -p "$LITEGS_CUDA_DIR"
  download_dir="$(mktemp -d)"
  for package in cuda-nvcc cuda-cudart cuda-cudart-dev cuda-cccl; do
    archive="$package-11.8.89-0.tar.bz2"
    wget -c -O "$download_dir/$archive" "https://conda.anaconda.org/nvidia/linux-64/$archive"
    tar -xjf "$download_dir/$archive" -C "$LITEGS_CUDA_DIR" --exclude=info
  done
fi

export CUDA_HOME="$LITEGS_CUDA_DIR"
export PATH="$LITEGS_ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS="${MAX_JOBS:-2}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

cd "$LITEGS_DIR"
"$UV_BIN" pip install --python "$PYTHON_BIN" --no-build-isolation \
  ./litegs/submodules/simple-knn
"$UV_BIN" pip install --python "$PYTHON_BIN" --no-build-isolation \
  ./litegs/submodules/fused_ssim
"$UV_BIN" pip install --python "$PYTHON_BIN" --no-build-isolation \
  ./litegs/submodules/gaussian_raster

echo "[完成] LiteGS 环境配置完成。"
echo "请显式指定数据集执行环境检查："
echo "  $PROJECT_DIR/scripts/04_litegs/check_environment.sh --dataset <COLMAP 数据目录>"
