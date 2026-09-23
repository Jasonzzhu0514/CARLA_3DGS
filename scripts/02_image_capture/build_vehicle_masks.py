#!/usr/bin/env python3
"""Build binary vehicle masks and masked RGB copies from CARLA raw semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# CARLA 0.9.16 CityObjectLabel.Car. Older tutorials often state that vehicles
# use tag 10; in the current label table 10 is Terrain and Car is 14.
VEHICLE_TAG = 14


def main() -> int:
    parser = argparse.ArgumentParser(description="从 CARLA 语义图生成车辆 mask。")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--vehicle-tag", type=int, default=VEHICLE_TAG)
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    images_dir = session_dir / "images"
    semantic_dir = session_dir / "semantic_raw"
    masks_dir = session_dir / "masks"
    masked_dir = session_dir / "images_masked"
    semantic_paths = sorted(semantic_dir.glob("*.png"))
    if not semantic_paths:
        raise SystemExit(f"找不到语义图: {semantic_dir}")

    masks_dir.mkdir(exist_ok=True)
    masked_dir.mkdir(exist_ok=True)

    first = np.asarray(Image.open(semantic_paths[0]).convert("RGBA"))
    channel_counts = [
        int(np.count_nonzero(first[:, :, channel] == args.vehicle_tag))
        for channel in range(3)
    ]
    tag_channel = int(np.argmax(channel_counts))
    if channel_counts[tag_channel] == 0:
        raise SystemExit(
            f"第一张语义图中没有车辆标签 {args.vehicle_tag}；"
            f"RGB 通道计数为 {channel_counts}"
        )

    records: list[dict[str, float | int | str]] = []
    for semantic_path in semantic_paths:
        rgb_path = images_dir / semantic_path.name
        if not rgb_path.is_file():
            raise SystemExit(f"缺少配对 RGB 图: {rgb_path}")

        semantic = np.asarray(Image.open(semantic_path).convert("RGBA"))
        mask_array = semantic[:, :, tag_channel] == args.vehicle_tag
        count = int(mask_array.sum())
        if count == 0:
            raise SystemExit(f"车辆 mask 为空: {semantic_path.name}")

        ys, xs = np.nonzero(mask_array)
        height, width = mask_array.shape
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)

        mask = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
        mask.save(masks_dir / semantic_path.name)

        rgb = np.asarray(Image.open(rgb_path).convert("RGB")).copy()
        rgb[~mask_array] = 0
        Image.fromarray(rgb, mode="RGB").save(masked_dir / semantic_path.name)

        records.append(
            {
                "filename": semantic_path.name,
                "vehicle_pixels": count,
                "image_fraction": count / (width * height),
                "bbox_width_fraction": bbox_width / width,
                "bbox_height_fraction": bbox_height / height,
            }
        )

    fractions = [float(record["image_fraction"]) for record in records]
    bbox_widths = [float(record["bbox_width_fraction"]) for record in records]
    bbox_heights = [float(record["bbox_height_fraction"]) for record in records]
    summary = {
        "vehicle_semantic_tag": args.vehicle_tag,
        "semantic_tag_channel": "RGBA"[tag_channel],
        "image_count": len(records),
        "vehicle_image_fraction": {
            "min": min(fractions),
            "mean": sum(fractions) / len(fractions),
            "max": max(fractions),
        },
        "vehicle_bbox_width_fraction": {
            "min": min(bbox_widths),
            "mean": sum(bbox_widths) / len(bbox_widths),
            "max": max(bbox_widths),
        },
        "vehicle_bbox_height_fraction": {
            "min": min(bbox_heights),
            "mean": sum(bbox_heights) / len(bbox_heights),
            "max": max(bbox_heights),
        },
        "frames": records,
    }
    (session_dir / "mask_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"车辆 mask: {len(records)} 张；标签通道={summary['semantic_tag_channel']}；"
        f"bbox 宽度占比 {min(bbox_widths):.1%}..{max(bbox_widths):.1%}，"
        f"均值 {sum(bbox_widths) / len(bbox_widths):.1%}"
    )
    print(f"二值 mask: {masks_dir}")
    print(f"黑背景预览/训练副本: {masked_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
