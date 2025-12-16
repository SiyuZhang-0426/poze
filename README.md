# Poze: Pi3-Guided Wan2.2 TI2V

Poze pairs Pi3 3D reconstruction with Wan2.2 text/image-to-video. Pi3 extracts 3D-aware latents from a reference image; Poze pools those latents into Wan’s context space so Wan can generate videos that stay consistent with the 3D structure.

## Model implementation (what Poze does)
- Pi3 runs on the input image and exposes decoder hidden states and patch metadata.
- `Pi3FeatureAdapter` reshapes decoder tokens back to a spatial grid, adaptively pools them to a small `target_tokens` grid, and projects them into Wan’s text-embedding dimension.
- The pooled tokens are sent to Wan TI2V as `extra_context`, giving Wan 3D-aware guidance alongside the text prompt.
- Pi3 stays frozen by default; Wan can be kept frozen for inference or marked trainable for finetuning. Poze also decodes Pi3 latents to return points/depth/RGB alongside the generated video.

## Environment
- Python ≥3.10, PyTorch ≥2.4 with CUDA.
- Install dependencies inside `pi3/` and `wan/`:
  ```bash
  cd pi3 && pip install -r requirements.txt
  cd ../wan && pip install -r requirements.txt
  ```

## Download checkpoints
`download_checkpoints.py` pulls both Pi3 and Wan2.2 TI2V-5B weights from Hugging Face (set `HF_TOKEN` or pass `--token`).
```bash
python download_checkpoints.py \
  --pi3-dir ./pi3_ckpt \
  --wan-dir ./Wan2.2-TI2V-5B \
  --token $HF_TOKEN
```
Options: `--pi3-id`, `--wan-id`, and `--revision` allow pointing to custom repos/tags.

## Inference
Generate a video with Pi3 guidance:
```bash
python inference.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --image frame0.png \
  --prompt "A drone flythrough of a canyon" \
  --adapter-tokens 64 \
  --device cuda \
  --offload-model true \
  --output outputs/canyon.mp4
```
`inference.py` saves the mp4 and can optionally dump latents (`--save-latents`) or Pi3 decodes (`--save-pi3`). `--frame-num` overrides the config’s default length.

## Finetune
Finetuning optimizes Wan while keeping Pi3 frozen. Provide at least one supervision signal (`--gt-video` or `--gt-points`).
```bash
python finetune.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --prompt "A walk through a forest" \
  --image ref.png \
  --gt-video data/target_video.pt \
  --steps 50 \
  --lr 1e-5 \
  --adapter-tokens 64 \
  --save-dir finetune_outputs
```
`finetune.py` uses MSE on video and L1 on Pi3 points (weighted by `--video-weight` / `--point-weight`), supports AMP, optional gradient clipping, and saves checkpoints at `--save-every` or on completion.
