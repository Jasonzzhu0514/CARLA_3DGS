# Stage 03: COLMAP sparse reconstruction

Create the isolated CUDA PyCOLMAP environment once:

```bash
./scripts/03_colmap/setup_gpu_environment.sh
```

Reconstruct a capture session by name or absolute path:

```bash
./scripts/03_colmap/run_sparse_reconstruction.sh --dataset <session>
```

The pipeline uses the CARLA `PINHOLE` intrinsics, CUDA SIFT extraction, and CUDA exhaustive
matching. Incremental mapping and bundle adjustment run on the CPU. Exhaustive matching provides
connections across orbit boundaries and camera heights.

After reconstruction, the script verifies model completeness and aligns the estimated camera
centers with the exact CARLA positions in `frames.jsonl`. The default acceptance thresholds are
0.10 m mean error and 0.25 m maximum error.

Outputs are written to `datasets/colmap/<session>/`; logs are written to `logs/<session>/colmap.log`.
Existing output directories are never overwritten.
