#!/usr/bin/env python3
"""Validate a text-exported COLMAP sparse model."""

from __future__ import annotations

import argparse
from pathlib import Path


def data_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 COLMAP 稀疏重建结果。")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--expected-images", type=int, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    text_dir = workspace / "sparse_txt"
    required = [
        workspace / "sparse" / "0" / "cameras.bin",
        workspace / "sparse" / "0" / "images.bin",
        workspace / "sparse" / "0" / "points3D.bin",
        workspace / "points3D.ply",
        text_dir / "cameras.txt",
        text_dir / "images.txt",
        text_dir / "points3D.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("COLMAP 检查失败，缺少文件：")
        for path in missing:
            print(f"- {path}")
        return 1

    camera_count = len(data_lines(text_dir / "cameras.txt"))
    image_lines = data_lines(text_dir / "images.txt")
    registered_images = len(image_lines[::2])
    point_count = len(data_lines(text_dir / "points3D.txt"))
    registration_rate = registered_images / args.expected_images

    print(f"相机数: {camera_count}")
    print(
        f"注册图像: {registered_images}/{args.expected_images} "
        f"({registration_rate:.1%})"
    )
    print(f"稀疏点数: {point_count}")

    if camera_count != 1:
        print("COLMAP 检查失败：预期所有图像共用一个相机内参。")
        return 1
    if registered_images < max(2, int(args.expected_images * 0.9)):
        print("COLMAP 检查失败：注册率低于 90%。")
        return 1
    if point_count == 0:
        print("COLMAP 检查失败：稀疏点云为空。")
        return 1

    print("COLMAP 稀疏模型检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
