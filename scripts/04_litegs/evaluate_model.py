#!/usr/bin/env python3
"""Render a trained LiteGS model on its COLMAP cameras and save metrics/previews."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import litegs
import litegs.config


def to_u8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def save_comparison(path: Path, gt: torch.Tensor, rendered: torch.Tensor, name: str) -> None:
    gt_u8 = to_u8(gt)
    rendered_u8 = to_u8(rendered)
    error_u8 = np.clip(np.abs(gt_u8.astype(np.int16) - rendered_u8.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    height, width, _ = gt_u8.shape
    header = 28
    canvas = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"ground truth: {name}", fill="black")
    draw.text((width + 8, 7), "LiteGS render", fill="black")
    draw.text((width * 2 + 8, 7), "absolute error x4", fill="black")
    canvas.paste(Image.fromarray(gt_u8), (0, header))
    canvas.paste(Image.fromarray(rendered_u8), (width, header))
    canvas.paste(Image.fromarray(error_u8), (width * 2, header))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--preview-stride", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.model / "evaluation"
    preview_dir = output_dir / "comparisons"
    preview_dir.mkdir(parents=True, exist_ok=True)

    lp, op, pp, _ = litegs.config.get_default_arg()
    lp.source_path = str(args.source)
    lp.images = "images"
    lp.model_path = str(args.model)
    lp.resolution = args.resolution
    lp.sh_degree = 3
    pp.device_preload = False

    cameras, frames, init_xyz, init_color = litegs.io_manager.load_colmap_result(
        lp.source_path, lp.images
    )
    del init_xyz, init_color
    dataset = litegs.data.CameraFrameDataset(cameras, frames, lp.resolution, False)

    ply_path = args.model / "point_cloud" / "finish" / "point_cloud.ply"
    xyz, scale, rot, sh_0, sh_rest, opacity = litegs.io_manager.load_ply(
        str(ply_path), lp.sh_degree
    )
    gaussian_count = int(xyz.shape[-1])
    tensors = [torch.from_numpy(value).float().cuda() for value in (xyz, scale, rot, sh_0, sh_rest, opacity)]
    del xyz, scale, rot, sh_0, sh_rest, opacity
    xyz_t, scale_t, rot_t, sh_0_t, sh_rest_t, opacity_t = litegs.scene.point.spatial_refine(
        False, None, *tensors
    )
    xyz_t, scale_t, rot_t, sh_0_t, sh_rest_t, opacity_t = litegs.scene.cluster.cluster_points(
        pp.cluster_size, xyz_t, scale_t, rot_t, sh_0_t, sh_rest_t, opacity_t
    )
    cluster_origin, cluster_extend = litegs.scene.cluster.get_cluster_AABB(
        xyz_t, scale_t.exp(), torch.nn.functional.normalize(rot_t, dim=0)
    )

    rows = []
    started = time.time()
    with torch.inference_mode():
        for index in range(len(dataset)):
            view, projection, frustum, gt = dataset[index]
            view = view.unsqueeze(0).cuda()
            projection = projection.unsqueeze(0).cuda()
            frustum = frustum.unsqueeze(0).cuda()
            gt = gt.unsqueeze(0).cuda().div_(255.0)
            _, cx, cs, cr, c0, crest, co = litegs.render.render_preprocess(
                cluster_origin,
                cluster_extend,
                frustum,
                xyz_t,
                scale_t,
                rot_t,
                sh_0_t,
                sh_rest_t,
                opacity_t,
                op,
                pp,
            )
            rendered, _, _, _ = litegs.render.render(
                view,
                projection,
                cx,
                cs,
                cr,
                c0,
                crest,
                co,
                lp.sh_degree,
                gt.shape[2:],
                pp,
            )
            if not torch.isfinite(rendered).all():
                raise RuntimeError(f"rendered image {index} contains NaN or Inf")
            mse = torch.mean((rendered - gt) ** 2)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            ssim = structural_similarity_index_measure(rendered, gt, data_range=1.0)
            rows.append(
                {
                    "index": index,
                    "image": frames[index].name,
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                }
            )
            if index % args.preview_stride == 0:
                save_comparison(preview_dir / f"{index:06d}.jpg", gt, rendered, frames[index].name)

    def summarize(items):
        return {
            "count": len(items),
            "psnr_mean": float(np.mean([row["psnr"] for row in items])),
            "psnr_min": float(np.min([row["psnr"] for row in items])),
            "ssim_mean": float(np.mean([row["ssim"] for row in items])),
            "ssim_min": float(np.min([row["ssim"] for row in items])),
        }

    block = math.ceil(len(rows) / 3)
    report = {
        "source": str(args.source),
        "model": str(args.model),
        "ply": str(ply_path),
        "gaussian_count": gaussian_count,
        "resolution_factor": args.resolution,
        "image_count": len(rows),
        "elapsed_seconds": time.time() - started,
        "scope": "All 360 views were used for training; these are training-view metrics.",
        "overall": summarize(rows),
        "height_blocks": {
            "low_0_119": summarize(rows[:block]),
            "middle_120_239": summarize(rows[block : 2 * block]),
            "high_240_359": summarize(rows[2 * block :]),
        },
        "per_image": rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
