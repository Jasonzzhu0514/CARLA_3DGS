# CARLA_3DGS

This repository provides a reproducible simulation pipeline for collecting multi-view vehicle
images in CARLA, estimating camera poses with COLMAP, and training a 3D Gaussian Splatting model
with LiteGS.

The workflow is split into four stages:

1. Start CARLA and verify vehicle spawning.
2. Capture synchronized RGB images, semantic labels, vehicle masks, camera intrinsics, and poses.
3. Run CUDA SIFT extraction and exhaustive matching, followed by COLMAP sparse reconstruction.
4. Check, train, and evaluate a LiteGS model.

## Repository layout

```text
.
├── README.md
├── configs/
│   ├── capture/                 # Versioned capture configurations
│   ├── requirements/            # Versioned Python dependency pins
│   └── paths.env.example        # Machine-local configuration template
└── scripts/
    ├── lib/                     # Shared configuration loader
    ├── 01_carla_basics/
    ├── 02_image_capture/
    ├── 03_colmap/
    └── 04_litegs/
```

Runtime datasets, generated models, and logs are excluded from version control.

## Requirements

- Linux with an NVIDIA GPU and compatible driver
- CARLA 0.9.16
- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- Git, GCC, and G++
- LiteGS dependencies installed by the provided setup script

The COLMAP setup uses `pycolmap-cuda12==4.2.0`. CUDA accelerates SIFT feature extraction and
exhaustive matching; incremental mapping and bundle adjustment run on the CPU.

## Configuration

Create a local configuration file and edit the paths for your machine:

```bash
cp configs/paths.env.example configs/paths.env
```

`configs/paths.env` is intentionally ignored by Git. Every shell entry point also supports the
`PROJECT_CONFIG_FILE` environment variable if the configuration is stored elsewhere:

```bash
PROJECT_CONFIG_FILE=/path/to/paths.env ./scripts/01_carla_basics/check_environment.sh
```

## Usage

Run commands from the repository root.

Check and start CARLA:

```bash
./scripts/01_carla_basics/check_environment.sh
./scripts/01_carla_basics/start_carla.sh
```

In another terminal, either spawn a vehicle for an interactive check:

```bash
./scripts/01_carla_basics/run_spawn_vehicle.sh
```

or collect a multi-height dataset:

```bash
./scripts/02_image_capture/run_capture.sh \
  vehicle_360views \
  configs/capture/capture_360_vehicle_focused.json
```

Create the CUDA PyCOLMAP environment once, then reconstruct a capture session:

```bash
./scripts/03_colmap/setup_gpu_environment.sh
./scripts/03_colmap/run_sparse_reconstruction.sh \
  --dataset <capture-session>
```

The reconstruction script validates file completeness and compares estimated camera centers with
the CARLA ground-truth positions saved during capture.

Set up LiteGS, run a short validation training, and start a full training run:

```bash
./scripts/04_litegs/setup_environment.sh
./scripts/04_litegs/run_smoke_train.sh --dataset <capture-session>
./scripts/04_litegs/run_full_train.sh --dataset <capture-session>
```

Evaluate a trained model:

```bash
./scripts/04_litegs/run_evaluation.sh \
  --dataset <capture-session> \
  --model outputs/<capture-session>/<training-run>
```

## Generated files

```text
datasets/raw/<session>/       # RGB, semantic labels, masks, intrinsics, CARLA poses
datasets/colmap/<session>/    # COLMAP database and sparse model
outputs/<session>/            # LiteGS checkpoints, PLY files, and evaluation renders
logs/<session>/               # Capture, reconstruction, and training logs
```

Capture and reconstruction scripts refuse to overwrite an existing session. Use a new session
name for each run so that inputs, outputs, and logs remain traceable.

## Current limitations

- LiteGS training currently uses RGB images only; captured vehicle masks are retained for future
  masked-loss experiments.
- Evaluation renders training-camera views. A separate held-out capture is required to measure
  novel-view generalization.

## License

This project's original code and documentation are licensed under the [MIT License](LICENSE).
Third-party software and dependencies, including CARLA and LiteGS, remain subject to their own licenses.
