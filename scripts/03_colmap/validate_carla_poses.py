#!/usr/bin/env python3
"""Compare COLMAP camera centers with exact CARLA capture positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit target ~= scale * source @ rotation + translation, allowing reflection."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    scale = np.sum((source_centered @ rotation) * target_centered) / np.sum(
        source_centered * source_centered
    )
    translation = target.mean(axis=0) - scale * source.mean(axis=0) @ rotation
    prediction = scale * source @ rotation + translation
    residual_m = np.linalg.norm(prediction - target, axis=1) / abs(scale)
    return float(scale), residual_m


def main() -> int:
    parser = argparse.ArgumentParser(description="用 CARLA 真值检查 COLMAP 相机中心。")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-mean-error-m", type=float, default=0.10)
    parser.add_argument("--max-single-error-m", type=float, default=0.25)
    args = parser.parse_args()

    frames_path = args.raw_dir / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text().splitlines()]
    rows.sort(key=lambda row: row["filename"])
    reconstruction = pycolmap.Reconstruction(args.model_dir)
    images = {image.name: image for image in reconstruction.images.values()}

    missing = [row["filename"] for row in rows if row["filename"] not in images]
    if missing:
        raise RuntimeError(f"COLMAP 缺少 {len(missing)} 个 CARLA 帧，例如 {missing[0]}")

    carla_centers = np.asarray(
        [
            [row["camera_transform"]["location_m"][axis] for axis in ("x", "y", "z")]
            for row in rows
        ],
        dtype=np.float64,
    )
    colmap_centers = np.asarray(
        [images[row["filename"]].projection_center() for row in rows],
        dtype=np.float64,
    )
    scale, errors = fit_similarity(carla_centers, colmap_centers)

    layer_results = []
    for layer in sorted({row["height_level_index"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["height_level_index"] == layer]
        layer_errors = errors[indices]
        layer_results.append(
            {
                "layer": layer,
                "images": len(indices),
                "mean_error_m": float(layer_errors.mean()),
                "max_error_m": float(layer_errors.max()),
            }
        )

    result = {
        "images": len(rows),
        "colmap_units_per_carla_meter": scale,
        "mean_camera_center_error_m": float(errors.mean()),
        "median_camera_center_error_m": float(np.median(errors)),
        "p95_camera_center_error_m": float(np.percentile(errors, 95)),
        "max_camera_center_error_m": float(errors.max()),
        "thresholds_m": {
            "mean": args.max_mean_error_m,
            "single": args.max_single_error_m,
        },
        "layers": layer_results,
    }
    output_path = args.model_dir.parent.parent / "carla_pose_validation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    print(
        "CARLA 位姿真值检查: "
        f"mean={errors.mean():.4f} m, median={np.median(errors):.4f} m, "
        f"p95={np.percentile(errors, 95):.4f} m, max={errors.max():.4f} m"
    )
    for layer in layer_results:
        print(
            f"  高度层 {layer['layer']}: mean={layer['mean_error_m']:.4f} m, "
            f"max={layer['max_error_m']:.4f} m"
        )

    if errors.mean() > args.max_mean_error_m or errors.max() > args.max_single_error_m:
        print("COLMAP 位姿检查失败：相机环与 CARLA 真值没有正确对齐。")
        return 1
    print("COLMAP 位姿检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
