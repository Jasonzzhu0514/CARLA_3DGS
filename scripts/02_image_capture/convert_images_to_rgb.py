#!/usr/bin/env python3
"""把某个采集 session 的 PNG 原地归一化为 8 位 RGB（去掉 alpha 通道）。

CARLA 的 ``image.save_to_disk()`` 永远写出 RGBA PNG，即使 alpha 通道处处为
255。LiteGS/3DGS 期望 3 通道输入，否则渲染结果（3 通道）与真值图像（4 通道）
会在训练第一步报错：

    RuntimeError: The size of tensor a (3) must match the size of tensor b (4)
    at non-singleton dimension 1

本工具会把 ``<session>/images/*.png`` 全部转换为 RGB 并原地覆盖。因为 alpha
恒为 255，该转换在信息上无损，只是去掉一个多余的通道。

需要 Pillow，所以要用装了 PIL 的解释器运行。CARLA 的 venv 没有 PIL，系统
``python3`` 有；也可以用 ``LITEGS_PYTHON_BIN``。

用法：

    python3 scripts/02_image_capture/convert_images_to_rgb.py datasets/raw/<session>
    python3 scripts/02_image_capture/convert_images_to_rgb.py datasets/raw/<session> --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 取决于运行解释器
    print(
        "需要 Pillow：请用带 PIL 的解释器运行，例如系统 python3 "
        "或 $LITEGS_PYTHON_BIN。",
        file=sys.stderr,
    )
    raise SystemExit(2)


def replace_with_rgb(path: Path, image: "Image.Image") -> None:
    """先写临时文件再原子替换，避免中途失败损坏原始 PNG。"""
    temp_path = path.with_name(f"{path.name}.rgb-tmp")
    try:
        image.save(temp_path, format="PNG")
        os.chmod(temp_path, path.stat().st_mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把采集 session 的 PNG 归一化为 8 位 RGB。"
    )
    parser.add_argument("session", type=Path, help="session 目录，例如 datasets/raw/<session>")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查不修改；发现非 RGB 图像时返回非零退出码。",
    )
    parser.add_argument("--quiet", action="store_true", help="只输出结论。")
    args = parser.parse_args()

    images_dir = args.session.resolve() / "images"
    if not images_dir.is_dir():
        print(f"找不到图像目录: {images_dir}", file=sys.stderr)
        return 2

    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        print(f"目录中没有 PNG: {images_dir}", file=sys.stderr)
        return 2

    already_rgb = 0
    converted = 0
    problems: list[tuple[str, str]] = []
    translucent: list[str] = []

    for path in paths:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            if mode == "RGB":
                already_rgb += 1
                continue
            if args.check:
                problems.append((path.name, mode))
                continue

            if "A" in mode:
                alpha_min, alpha_max = image.getchannel("A").getextrema()
                if alpha_min != 255 or alpha_max != 255:
                    translucent.append(f"{path.name} (alpha {alpha_min}..{alpha_max})")

            rgb_image = image.convert("RGB")

        replace_with_rgb(path, rgb_image)
        converted += 1

    if args.check:
        print(f"图像总数: {len(paths)}，已是 RGB: {already_rgb}，非 RGB: {len(problems)}")
        if problems:
            print("非 RGB 图像（文件, 当前模式）：")
            for name, mode in problems[:20]:
                print(f"- {name}: {mode}")
            if len(problems) > 20:
                print(f"- 其余 {len(problems) - 20} 张省略")
            return 1
        return 0

    if not args.quiet:
        print(f"图像目录: {images_dir}")
    print(f"图像总数: {len(paths)}，本次转换: {converted}，原本已是 RGB: {already_rgb}")
    if translucent:
        print("警告：以下图像 alpha 并非全 255，去 alpha 会丢失信息：", file=sys.stderr)
        for item in translucent[:20]:
            print(f"- {item}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
