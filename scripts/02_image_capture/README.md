# Stage 02: multi-view capture

The capture stage spawns a stationary vehicle and synchronized RGB and semantic cameras. It saves
RGB images, optional semantic labels and masks, camera intrinsics, exact CARLA poses, and session
metadata.

CARLA must already be running. The session label and configuration are both required. Run the
multi-height 360-view configuration with:

```bash
./scripts/02_image_capture/run_capture.sh \
  vehicle_360views \
  configs/capture/capture_360_vehicle_focused.json
```

Each run creates a timestamped directory under `datasets/raw/` and writes a matching log directory
under `logs/`. The script refuses to overwrite an existing session.

CARLA writes RGBA PNG files. `run_capture.sh` automatically removes the constant alpha channel so
that downstream training receives three-channel RGB images. When semantic capture is enabled, it
also creates binary vehicle masks.

Validate or convert an existing session manually:

```bash
source configs/paths.env
"$CARLA_PYTHON_BIN" scripts/02_image_capture/validate_capture.py datasets/raw/<session>
python3 scripts/02_image_capture/convert_images_to_rgb.py datasets/raw/<session> --check
```

The output layout is:

```text
datasets/raw/<session>/
├── images/
├── semantic_raw/          # Optional
├── masks/                 # Optional
├── images_masked/         # Optional inspection copies
├── camera_intrinsics.json
├── frames.jsonl
└── session.json
```
