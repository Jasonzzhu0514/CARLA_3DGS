#!/usr/bin/env python3
"""Run CUDA SIFT extraction/exhaustive matching and sparse mapping."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 CUDA PyCOLMAP 生成稀疏模型。")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", default="0")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    work_dir = args.work_dir.resolve()
    image_dir = work_dir / "images"
    database_path = work_dir / "database.db"
    intrinsics = json.loads(
        (raw_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )

    if not pycolmap.has_cuda:
        raise RuntimeError("当前 PyCOLMAP 未启用 CUDA，已停止以避免静默退回 CPU")

    camera_params = ",".join(
        str(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")
    )
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = camera_params

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = True
    extraction_options.gpu_index = args.gpu_index
    extraction_options.max_image_size = -1
    extraction_options.sift.max_num_features = 2048

    print(
        f"[GPU] PyCOLMAP {pycolmap.__version__}, CUDA={pycolmap.has_cuda}, "
        f"gpu_index={args.gpu_index}",
        flush=True,
    )
    print("[1/3] CUDA SIFT 特征提取", flush=True)
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = True
    matching_options.gpu_index = args.gpu_index
    pairing_options = pycolmap.ExhaustivePairingOptions()
    pairing_options.block_size = 50

    print("[2/3] CUDA SIFT 全匹配", flush=True)
    pycolmap.match_exhaustive(
        database_path,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cuda,
    )

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.ba_refine_focal_length = False
    mapping_options.ba_refine_principal_point = False
    mapping_options.ba_refine_extra_params = False
    mapping_options.mapper.abs_pose_refine_focal_length = False
    mapping_options.mapper.abs_pose_refine_extra_params = False
    # CUDA wheel guarantees GPU SIFT. Bundle adjustment support depends on how
    # Ceres was built, so keep mapping/BA on CPU instead of allowing fallback.
    mapping_options.ba_use_gpu = False

    print("[3/3] 增量建图与 CPU bundle adjustment", flush=True)
    with tempfile.TemporaryDirectory(prefix="mapping-", dir=work_dir) as temp_dir:
        reconstructions = pycolmap.incremental_mapping(
            database_path,
            image_dir,
            temp_dir,
            options=mapping_options,
        )
        if not reconstructions:
            raise RuntimeError("COLMAP 没有生成稀疏模型")
        reconstruction = max(
            reconstructions.values(),
            key=lambda model: (model.num_reg_images(), model.num_points3D()),
        )

    model_dir = work_dir / "sparse" / "0"
    text_dir = work_dir / "sparse_txt"
    model_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(model_dir)
    reconstruction.write_text(text_dir)
    reconstruction.export_PLY(work_dir / "points3D.ply")

    summary = reconstruction.summary()
    (work_dir / "model_analyzer.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    backend = {
        "pycolmap_version": pycolmap.__version__,
        "has_cuda": bool(pycolmap.has_cuda),
        "gpu_index": args.gpu_index,
        "gpu_stages": [
            "SIFT feature extraction",
            "SIFT exhaustive feature matching",
        ],
        "cpu_stages": ["incremental mapping", "bundle adjustment"],
        "camera_model": "PINHOLE",
        "camera_params": camera_params,
    }
    (work_dir / "colmap_backend.json").write_text(
        json.dumps(backend, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(
        f"选中模型: 注册 {reconstruction.num_reg_images()} 张，"
        f"稀疏点 {reconstruction.num_points3D()}，"
        f"平均重投影误差 {reconstruction.compute_mean_reprojection_error():.6f}px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
