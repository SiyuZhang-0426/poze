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
- `scripts/pozeinference.sh`: Slurm+Apptainer example that wraps `inference.py` with Pi3 guidance. Edit the image path, prompt, checkpoint locations, and cluster settings, then submit via `sbatch scripts/pozeinference.sh` (or run the python command inside the script directly on a workstation).
- `scripts/pozefinetune.sh`: Slurm+Apptainer entrypoint for finetuning on a preprocessed 4DNeX dataset. It calls `finetune.py` with `--dataset-root` pointing at the Wan-style dataset directory (for example `./data/wan21`).

## Code layout
- Shared pipeline helpers (model setup, timestep model selection, Pi3 adapter utilities) live in `wan/wan/utils/pipeline_utils.py` and are used by `textimage2video.py`, `text2video.py`, and `image2video.py` to keep those entrypoints leaner.

### Visualize PLY point clouds
Render a quick mp4/gif preview from a single `.ply` file or a directory/glob of PLY frames (requires `matplotlib`, `plyfile`, and `imageio[ffmpeg]`):
```bash
python visualize_ply_sequence.py \
  --input outputs/pi3_points \
  --save outputs/pi3_points_preview.mp4 \
  --max-points 60000 \
  --fps 6
```

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
- `--save-pi3`: directory (or legacy `.ply` path) to save Pi3 decodes as per-frame PLYs named `<image_stem>_0000.ply`, `<image_stem>_0001.ply`, etc.
- `--frame-num`: override the config’s default video length.
- `--offload-model`: optional; pass `true` or `false` (common boolean strings like `yes/no/1/0` also work) to control Wan CPU offloading. If omitted, it defaults to true.
- `--wan-config`: choose another Wan config key (defaults to `ti2v-5B`).
- `--use-pi3`: set false to disable Pi3 conditioning (runs plain Wan TI2V).
- `--concat-method`: fuse Pi3 latents with RGB latents by channel (default), frame (time-wise concat), width (side-by-side, doubling output width), or height (stacked vertically, doubling output height).
- `--pi3-checkpoint` / `--pi3-pretrained-id`: point to local Pi3 weights or a HF id.
- `--output`: override the auto-named mp4 path under `outputs/`.

Another inference example with extra outputs and fixed length:
```bash
python inference.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --image ./data/ref.png \
  --prompt "A robot walking through neon streets" \
  --frame-num 32 \
  --offload-model true \
  --save-latents outputs/robot_latents.pt \
  --save-pi3 outputs/robot_pi3_points \
  --output outputs/robot.mp4
```

## Finetune
Finetuning optimizes Wan while keeping Pi3 frozen and only updates the Wan TI2V backbone. Provide at least one supervision signal (`--gt-video` or `--gt-points` in single-example mode, or dataset latents/videos in dataset mode).
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
- Keep `--offload-model` false for training (offloading slows optimization); use `--amp true` (default) for lower VRAM, and `--max-grad-norm` to clip gradients.
- Override length with `--frame-num`; choose a different config via `--wan-config`.
- `--log-every` controls logging frequency; `--save-every` >0 writes intermediate Wan checkpoints to `--save-dir`.

Finetune example with point supervision, AMP, and periodic checkpoints:
```bash
python finetune.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --prompt "A walk through a forest" \
  --image ref.png \
  --gt-points data/points.pt \
  --steps 100 \
  --lr 5e-6 \
  --video-weight 1.0 \
  --point-weight 0.5 \
  --amp true \
  --max-grad-norm 1.0 \
  --save-every 20 \
  --save-dir finetune_outputs/forest_run
```

To finetune directly on the preprocessed 4DNeX dataset (built via `4DNeX/build_wan_dataset.py`), point the script at the dataset root. Cached Wan VAE latents are used when available, while Pi3 remains frozen and Wan trains end-to-end:
```bash
python finetune.py \
  --wan-ckpt-dir ./Wan2.2-TI2V-5B \
  --dataset-root ./data/wan21 \
  --dataset-use-latents true \
  --batch-size 1 \
  --steps 200 \
  --latent-weight 1.0 \
  --video-weight 0.5
```
`--prompt` and `--image` are not needed in dataset mode; prompts, first-frame images (`first_frames/<stem>.png`), and cached latents (`cache/video_latent/wan-i2v/<resolution>/<stem>.safetensors`) are discovered from the dataset directory. When `--dataset-use-latents false` or when a sample has no latent file, the RGB video under `videos/` is loaded and used for supervision instead.

4DNeX dataset preparation follows `4DNeX/build_wan_dataset.py`. After downloading the raw 4DNeX data into `./data`, run:
```bash
python 4DNeX/build_wan_dataset.py \
  --data_dir ./data \
  --out ./data/wan21
```
This produces a Wan-style directory with:
- `prompts.txt` and `videos.txt` listing captions and video paths
- `first_frames/` with first-frame PNGs per video
- `cache/video_latent/wan-i2v/<resolution>/` containing Wan VAE latents (`encoded_video`)

In dataset mode the most relevant configuration flags are:
- `--dataset-root`: path to the preprocessed 4DNeX dataset (for example `./data/wan21`)
- `--dataset-prompt-file`, `--dataset-video-list`, `--dataset-image-dir`: override the default prompt/video/image filenames if needed
- `--dataset-latent-dir`: optional explicit directory with Wan VAE latents; when omitted, the script discovers the first `cache/video_latent/wan-i2v/*` subdirectory under `--dataset-root`
- `--dataset-use-latents`: when true, uses cached Wan VAE latents if available; when false or when a sample has no latent file, the video is decoded and used for supervision instead

During training the script logs the total loss and aggregates the video MSE into a simple PSNR metric computed from the video reconstruction loss; higher PSNR indicates better reconstruction quality. Average losses across all steps are reported at the end of finetuning, and intermediate/final Wan checkpoints are written to `--save-dir` when `--save-every` is greater than zero.
