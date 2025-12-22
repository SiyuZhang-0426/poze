# Poze: Pi3-Guided Wan2.2 TI2V

Poze (project name) pairs Pi3 3D reconstruction with Wan2.2 text/image-to-video. Pi3 extracts 3D-aware latents from a reference image, and Poze pools those latents into Wan’s context space so Wan can generate videos that stay consistent with the 3D structure.

## Model implementation (what Poze does)
- Pi3 runs on the input image and exposes decoder hidden states and patch metadata.
- The Pi3 decoder tokens are reshaped back to a spatial grid, resized to Wan’s latent resolution, and concatenated channel-wise with the encoded RGB latents before diffusion (no `extra_context` tokens).
- Wan receives this fused latent volume together with the text prompt to generate a temporally consistent video.
- Pi3 stays frozen by default; Wan can be kept frozen for inference or marked trainable for finetuning. Poze also decodes Pi3 latents to return points/depth/RGB alongside the generated video.
- Pi3 latents are fed through a 1×1×1 adapter seeded as an identity copy of the RGB channels (RGB weights are cloned into the Pi3 channels so fusion starts as a passthrough), and Wan’s patch embedding duplicates the RGB weights for the Pi3 half to avoid random init jitter.
 - Inputs are padded to the Pi3 patch size, and decoder tokens are reshaped to `(B, C, F, H, W)` before being fused with Wan latents—this padding/reshaping resolves the Pi3-vs-Wan shape mismatch.

## Environment
- Python ≥3.10, PyTorch ≥2.4 with CUDA.
- Install dependencies from `pi3/` and `wan/` (run from the repo root):
  ```bash
  pip install -r pi3/requirements.txt
  pip install -r wan/requirements.txt
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

## Scripts
- `scripts/hf_download.sh`: downloads Pi3 and Wan checkpoints using the lowercase `hf_token` environment variable (run `export hf_token=$HF_TOKEN` if you already have an uppercase token set). Run from repo root with `bash scripts/hf_download.sh`.
- `scripts/pi3inference.sh`: Slurm+Apptainer example for the upstream Pi3 demo (`pi3/example.py`). Update the project path, `--data-path`, and partition (`$vp`) to match your cluster, then submit with `sbatch scripts/pi3inference.sh`.
- `scripts/pozeinference.sh`: Slurm+Apptainer example that wraps `inference.py` with Pi3 guidance. Edit the image path, prompt, checkpoint locations, and cluster settings, then submit via `sbatch scripts/pozeinference.sh`.
- `scripts/pozefinetune.sh`: empty placeholder for a similar Slurm entrypoint to finetune; duplicate `pozeinference.sh` as a starting template (or mirror its commands) and fill in your dataset/paths.

## Inference
Generate a video with Pi3 guidance:
```bash
python inference.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --image frame0.png \
  --prompt "A drone flythrough of a canyon" \
  --device cuda \
  --output outputs/canyon.mp4
```
`inference.py` saves the output as an mp4. Helpful flags:
- `--save-latents`: dump latents.
- `--save-pi3`: save Pi3 decodes.
- `--frame-num`: override the config’s default video length.
- `--offload-model`: optional; pass `true` or `false` (common boolean strings like `yes/no/1/0` also work) to control Wan CPU offloading. If omitted, it defaults to true.

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
  --save-dir finetune_outputs
```
Ground-truth tensors should match the script expectations: video as a torch tensor shaped `(C, F, H, W)` (C=channels, F=frames, H=height, W=width) and Pi3 points as `(B, N, H, W, 3)` (B=batch, N=points, last `3` for XYZ) saved via `.pt`/`.pth`.
`finetune.py` details:
- Uses MSE on video and L1 on Pi3 points, weighted by `--video-weight` and `--point-weight`.
- Supports automatic mixed precision (AMP) and optional gradient clipping.
- Saves checkpoints at `--save-every` intervals or on completion.
