#!/usr/bin/env python3
"""Validate file counts, JSON metadata, and basic PNG headers for a capture."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def png_header(path: Path) -> tuple[int, int, int, int]:
    """读取 IHDR：宽、高、位深、颜色类型。"""
    with path.open("rb") as stream:
        header = stream.read(26)
    if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是有效的 PNG 文件: {path}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth = header[24]
    color_type = header[25]
    return width, height, bit_depth, color_type


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 CARLA 环绕采集结果。")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    intrinsics = json.loads(
        (session_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    frames = [
        json.loads(line)
        for line in (session_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    images = sorted((session_dir / "images").glob("*.png"))
    semantic_images = sorted((session_dir / "semantic_raw").glob("*.png"))
    masks = sorted((session_dir / "masks").glob("*.png"))
    masked_images = sorted((session_dir / "images_masked").glob("*.png"))

    config = session["config"]
    if "camera_height_levels_m" in config:
        expected = int(config["views_per_height"]) * len(
            config["camera_height_levels_m"]
        )
    else:
        expected = int(config["views"])
    errors: list[str] = []
    if session.get("status") != "complete":
        errors.append(f"session 状态不是 complete: {session.get('status')}")
    if len(images) != expected:
        errors.append(f"PNG 数量 {len(images)} != 期望值 {expected}")
    if len(frames) != expected:
        errors.append(f"元数据数量 {len(frames)} != 期望值 {expected}")
    if session.get("captured_images") != expected:
        errors.append("session.json 的 captured_images 不正确")
    expects_semantic = bool(config.get("capture_semantic_vehicle_mask", False))
    if expects_semantic:
        for label, paths in (
            ("语义图", semantic_images),
            ("车辆 mask", masks),
            ("masked RGB", masked_images),
        ):
            if len(paths) != expected:
                errors.append(f"{label}数量 {len(paths)} != 期望值 {expected}")
        if session.get("captured_semantic_images") != expected:
            errors.append("session.json 的 captured_semantic_images 不正确")

    if "camera_height_levels_m" in config and len(frames) == expected:
        expected_per_level = int(config["views_per_height"])
        for level_index in range(len(config["camera_height_levels_m"])):
            actual = sum(
                frame.get("height_level_index") == level_index for frame in frames
            )
            if actual != expected_per_level:
                errors.append(
                    f"高度层 {level_index} 的图像数量 {actual} != {expected_per_level}"
                )

    expected_size = (int(intrinsics["width"]), int(intrinsics["height"]))
    frame_ids = []
    for index, frame in enumerate(frames):
        image_path = session_dir / "images" / frame["filename"]
        if not image_path.is_file():
            errors.append(f"缺少图像: {image_path.name}")
            continue
        try:
            width, height, bit_depth, color_type = png_header(image_path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if (width, height) != expected_size:
            errors.append(f"{image_path.name} 尺寸 {(width, height)} != {expected_size}")
        if (bit_depth, color_type) != (8, 2):
            errors.append(
                f"{image_path.name} 不是 8 位 RGB "
                f"(bit_depth={bit_depth}, color_type={color_type})；"
                "请运行 convert_images_to_rgb.py 去掉 alpha 通道"
            )
        if frame["index"] != index:
            errors.append(f"第 {index} 条元数据的 index 不正确")
        frame_ids.append(int(frame["frame"]))
        if expects_semantic and frame.get("semantic_frame") != frame.get("frame"):
            errors.append(f"{image_path.name} 的 RGB/语义 frame 不同步")

    if frame_ids != sorted(set(frame_ids)):
        errors.append("frame ID 不唯一或没有严格递增")

    if errors:
        print("采集结果检查失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"检查通过: {session_dir}")
    print(f"图像: {len(images)} 张，尺寸: {expected_size[0]}x{expected_size[1]}，格式: 8 位 RGB")
    print(f"frame 范围: {frame_ids[0]}..{frame_ids[-1]}")
    print(f"相机模型: {intrinsics['model']}，fx={intrinsics['fx']:.3f}")
    if expects_semantic:
        print(f"语义图、车辆 mask、masked RGB: 各 {len(masks)} 张，均与 RGB 同步")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
