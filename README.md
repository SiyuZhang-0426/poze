# Pi3-Guided Wan2.2 TI2V Quickstart

This repository combines **Pi3** (3D reconstruction) with **Wan2.2** (text/image-to-video) so Pi3 latents can guide Wan’s TI2V generation and let you compute losses in both video and 3D space.

## What was added
- Pi3 exposes decoder latents and can decode from saved latents.
- Wan TI2V accepts optional `extra_context` tokens and can run with gradients enabled.
- `Pi3GuidedTI2V` adapter pools Pi3 latents, maps them to Wan context space, and drives Wan generation; Pi3 stays frozen while Wan can be finetuned.

## Environment
- Python ≥3.10, PyTorch ≥2.4 with CUDA.
- Install Pi3 and Wan requirements (from their subfolders) including FlashAttention and VAE dependencies.

```bash
# Pi3 deps
cd pi3 && pip install -r requirements.txt
# Wan deps
cd ../wan && pip install -r requirements.txt
```

## Inference example
```python
from wan import configs
from wan.pi3_guidance import Pi3GuidedTI2V

guide = Pi3GuidedTI2V(
    wan_config=configs.WAN_CONFIGS["ti2v-5B"],
    wan_checkpoint_dir="/path/to/wan_ckpts",   # contains Wan2.2 TI2V weights
    pi3_pretrained_id="yyfz233/Pi3",           # HF id for Pi3
)

out = guide.generate_with_3d(
    prompt="A drone flythrough of a canyon",
    image="frame0.png",
    frame_num=81,
)

video = out["video"]                 # (C, F, H, W) tensor
pointcloud = out["pi3"]["points"]    # (B, N, H, W, 3) world-space points
```

## Finetuning hook (optional)
Set `enable_grad=True` in `generate_with_3d` to allow Wan gradients; Pi3 stays frozen. You can compute custom losses:

```python
out = guide.training_step(
    prompt="A walk through a forest",
    image="ref.png",
    gt_video=target_video_tensor,     # optional
    gt_points=target_pointcloud,      # optional
    video_weight=1.0,
    point_weight=1.0,
    frame_num=81,
    enable_grad=True,
)
out["loss"].backward()
```

## Notes
- Checkpoints: pass `wan_checkpoint_dir` for Wan weights. For Pi3, either rely on `pi3_pretrained_id` or provide `pi3_checkpoint`.
- Safety: custom checkpoints load with `weights_only=True` by default; disable only if the format requires pickled metadata.
- GPU memory: set `offload_model=True` (default) to reduce VRAM; disable for finetuning to keep gradients resident.
