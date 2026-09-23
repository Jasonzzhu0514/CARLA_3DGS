# Stage 04: LiteGS training and evaluation

Create the pinned LiteGS environment and compile its CUDA extensions:

```bash
./scripts/04_litegs/setup_environment.sh
```

All data commands require an explicit COLMAP session name or path:

```bash
./scripts/04_litegs/check_environment.sh --dataset <session>
./scripts/04_litegs/run_smoke_train.sh --dataset <session>
./scripts/04_litegs/run_full_train.sh --dataset <session>
```

The smoke run performs 720 image iterations without densification. The full run performs 30,000
image iterations at half input resolution. Both create timestamped output and log directories.

Evaluate a trained model with:

```bash
./scripts/04_litegs/run_evaluation.sh \
  --dataset <session> \
  --model outputs/<session>/<training-run>
```

The current trainer consumes RGB images. Captured masks are preserved but are not yet applied to
the LiteGS loss.
