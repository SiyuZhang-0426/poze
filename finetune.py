import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _add_repo_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    for rel in ("wan", "pi3"):
        candidate = repo_root / rel
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    return repo_root


REPO_ROOT = _add_repo_to_path()

from wan import configs
from wan.pi3_guidance import Pi3GuidedTI2V
from wan.utils.utils import str2bool


@dataclass
class DatasetSample:
    prompt: str
    image_path: Path
    video_path: Optional[Path]
    latent_path: Optional[Path]


def _load_tensor(path: Path) -> torch.Tensor:
    ext = path.suffix.lower()
    if ext in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict) and "tensor" in obj:
            return obj["tensor"]
        raise ValueError(f"Unsupported tensor format in {path}")
    if ext == ".safetensors":
        data = load_file(path)
        if "encoded_video" in data:
            return data["encoded_video"]
        if len(data) == 1:
            return next(iter(data.values()))
        raise ValueError(f"Safetensors file {path} missing 'encoded_video' key")
    raise ValueError(f"Unsupported tensor extension for {path}")


def _load_single_video(path: Path) -> torch.Tensor:
    import imageio.v2 as imageio
    import numpy as np

    frames_np: List[torch.Tensor] = []
    reader = imageio.get_reader(path.as_posix())
    for frame in reader:
        frame_tensor = torch.from_numpy(frame).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frames_np.append(frame_tensor)
    if not frames_np:
        raise RuntimeError(f"No frames read from {path}")
    video = torch.stack(frames_np, dim=1)
    video = video / 127.5 - 1.0
    return video


def _discover_latent_dir(dataset_root: Path) -> Optional[Path]:
    base = dataset_root / "cache" / "video_latent" / "wan-i2v"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def _build_dataset(
    dataset_root: Path,
    prompt_file: str,
    video_list_file: str,
    image_dir: str,
    latent_dir: Optional[Path],
) -> List[DatasetSample]:
    prompt_path = dataset_root / prompt_file
    video_list_path = dataset_root / video_list_file
    image_root = dataset_root / image_dir

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    with open(video_list_path, "r", encoding="utf-8") as f:
        rel_videos = [line.strip() for line in f.readlines() if line.strip()]

    if len(prompts) != len(rel_videos):
        raise ValueError(
            f"Prompt count ({len(prompts)}) and video list count ({len(rel_videos)}) do not match."
        )

    samples: List[DatasetSample] = []
    for prompt, rel in zip(prompts, rel_videos):
        video_path = dataset_root / rel
        stem = video_path.stem
        image_path = image_root / f"{stem}.png"
        latent_path = None
        if latent_dir is not None:
            candidate = latent_dir / f"{stem}.safetensors"
            if candidate.exists():
                latent_path = candidate
        samples.append(
            DatasetSample(
                prompt=prompt,
                image_path=image_path,
                video_path=video_path if video_path.exists() else None,
                latent_path=latent_path,
            )
        )
    return samples


def _psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(4.0 / mse)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune Wan2.2 TI2V with optional Pi3 guidance."
    )

    parser.add_argument(
        "--wan-ckpt-dir",
        required=True,
        help="Directory containing Wan2.2 TI2V checkpoints.",
    )
    parser.add_argument(
        "--wan-config",
        default="ti2v-5B",
        choices=list(configs.WAN_CONFIGS.keys()),
        help="Wan TI2V config key to load.",
    )
    parser.add_argument(
        "--pi3-checkpoint",
        default=None,
        help="Optional local Pi3 checkpoint path; if omitted, pulls the pretrained id.",
    )
    parser.add_argument(
        "--pi3-pretrained-id",
        default="yyfz233/Pi3",
        help="HuggingFace model id for Pi3 when no checkpoint is provided.",
    )
    parser.add_argument(
        "--use-pi3",
        type=str2bool,
        default=True,
        help="Enable Pi3-guided conditioning; set false to run Wan TI2V without Pi3.",
    )
    parser.add_argument(
        "--concat-method",
        default="channel",
        choices=("channel", "frame", "width", "height"),
        help="How to fuse Pi3 latents with RGB latents.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to Pi3GuidedTI2V, e.g., cuda or cpu.",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=None,
        help="Number of frames to generate. Defaults to config.frame_num.",
    )
    parser.add_argument(
        "--offload-model",
        type=str2bool,
        default=False,
        help="Offload Wan to CPU between steps. Keep false for fastest training.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Effective batch size per optimization step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for Wan optimizer.",
    )
    parser.add_argument(
        "--video-weight",
        type=float,
        default=1.0,
        help="Loss weight for video reconstruction.",
    )
    parser.add_argument(
        "--latent-weight",
        type=float,
        default=0.0,
        help="Loss weight for latent supervision (dataset mode).",
    )
    parser.add_argument(
        "--point-weight",
        type=float,
        default=0.0,
        help="Loss weight for Pi3 point supervision.",
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Enable automatic mixed precision (AMP).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Max gradient norm for clipping; 0 disables clipping.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save Wan checkpoints every N steps; 0 disables intermediate saves.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "finetune_outputs",
        help="Directory where Wan checkpoints are saved.",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt for single-example finetuning.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Reference image for single-example finetuning.",
    )
    parser.add_argument(
        "--gt-video",
        type=Path,
        default=None,
        help="Ground truth video tensor (.pt/.pth) shaped (C, F, H, W).",
    )
    parser.add_argument(
        "--gt-points",
        type=Path,
        default=None,
        help="Ground truth Pi3 points tensor (.pt/.pth) shaped (B, F, H, W, 3).",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory of preprocessed 4DNeX Wan-style dataset (e.g., ./data/wan21).",
    )
    parser.add_argument(
        "--dataset-prompt-file",
        default="prompts.txt",
        help="Name of prompt file inside dataset root.",
    )
    parser.add_argument(
        "--dataset-video-list",
        default="videos.txt",
        help="Name of video list file inside dataset root.",
    )
    parser.add_argument(
        "--dataset-image-dir",
        default="first_frames",
        help="Subdirectory containing first-frame images.",
    )
    parser.add_argument(
        "--dataset-latent-dir",
        type=Path,
        default=None,
        help="Optional explicit directory of Wan VAE latents; when omitted, auto-discovers under cache/video_latent/wan-i2v.",
    )
    parser.add_argument(
        "--dataset-use-latents",
        type=str2bool,
        default=True,
        help="Use cached Wan VAE latents when available; otherwise load RGB videos.",
    )

    args = parser.parse_args()

    if args.dataset_root is None:
        if args.prompt is None or args.image is None:
            parser.error(
                "Single-example mode requires --prompt and --image when --dataset-root is not provided."
            )
        if args.gt_video is None and args.gt_points is None:
            parser.error(
                "Provide at least one supervision signal: --gt-video or --gt-points."
            )
    return args


def _setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_guide(args: argparse.Namespace) -> Tuple[Pi3GuidedTI2V, int]:
    cfg = configs.WAN_CONFIGS[args.wan_config]
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    guide = Pi3GuidedTI2V(
        wan_config=cfg,
        wan_checkpoint_dir=args.wan_ckpt_dir,
        pi3_checkpoint=args.pi3_checkpoint,
        pi3_pretrained_id=args.pi3_pretrained_id,
        device=args.device,
        trainable_wan=True,
        use_pi3=args.use_pi3,
        concat_method=args.concat_method,
    )
    return guide, frame_num


def _optimizer_parameters(guide: Pi3GuidedTI2V):
    return guide.wan.model.parameters()


def _train_single_example(args: argparse.Namespace) -> None:
    guide, frame_num = _build_guide(args)
    device = torch.device(args.device)

    gt_video = None
    if args.gt_video is not None:
        gt_video = _load_tensor(args.gt_video).to(device)
    gt_points = None
    if args.gt_points is not None:
        gt_points = _load_tensor(args.gt_points).to(device)

    optimizer = torch.optim.AdamW(
        _optimizer_parameters(guide),
        lr=args.lr,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    running_loss = 0.0
    running_video_mse = 0.0
    running_steps = 0

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
            outputs = guide.training_step(
                prompt=args.prompt,
                image=args.image,
                gt_video=gt_video,
                gt_points=gt_points,
                gt_latent=None,
                video_weight=args.video_weight,
                latent_weight=args.latent_weight,
                point_weight=args.point_weight,
                frame_num=frame_num,
                offload_model=args.offload_model,
            )
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        if args.max_grad_norm and args.max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                _optimizer_parameters(guide), args.max_grad_norm
            )
        scaler.step(optimizer)
        scaler.update()

        loss_scalar = loss.detach().float().item()
        running_loss += loss_scalar
        if "video" in outputs["losses"]:
            video_mse = outputs["losses"]["video"].detach().float().item()
            running_video_mse += video_mse
        running_steps += 1

        if step % args.log_every == 0 or step == args.steps:
            avg_loss = running_loss / max(running_steps, 1)
            avg_mse = running_video_mse / max(running_steps, 1e-8)
            psnr = _psnr(avg_mse) if running_video_mse > 0 else float("nan")
            logging.info(
                "Step %d/%d | loss=%.4f | video_mse=%.4f | psnr=%.2f dB",
                step,
                args.steps,
                avg_loss,
                avg_mse,
                psnr,
            )
            running_loss = 0.0
            running_video_mse = 0.0
            running_steps = 0

        if args.save_every and args.save_every > 0 and step % args.save_every == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = args.save_dir / f"wan_step_{step:06d}.pt"
            torch.save(guide.wan.model.state_dict(), ckpt_path)
            logging.info("Saved intermediate Wan checkpoint to %s", ckpt_path)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = args.save_dir / "wan_finetuned.pt"
    torch.save(guide.wan.model.state_dict(), final_ckpt)
    logging.info("Saved final Wan checkpoint to %s", final_ckpt)


def _train_dataset(args: argparse.Namespace) -> None:
    if args.dataset_root is None:
        raise ValueError("Dataset mode requires --dataset-root.")

    guide, frame_num = _build_guide(args)
    device = torch.device(args.device)

    latent_dir = args.dataset_latent_dir
    if latent_dir is None:
        latent_dir = _discover_latent_dir(args.dataset_root)

    samples = _build_dataset(
        dataset_root=args.dataset_root,
        prompt_file=args.dataset_prompt_file,
        video_list_file=args.dataset_video_list,
        image_dir=args.dataset_image_dir,
        latent_dir=latent_dir,
    )
    if not samples:
        raise ValueError(f"No samples found under dataset root {args.dataset_root}")

    optimizer = torch.optim.AdamW(
        _optimizer_parameters(guide),
        lr=args.lr,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    running_loss = 0.0
    running_video_mse = 0.0
    running_steps = 0

    step = 0
    sample_index = 0
    while step < args.steps:
        batch_prompts: List[str] = []
        batch_images: List[str] = []
        batch_gt_videos: List[Optional[torch.Tensor]] = []
        batch_gt_latents: List[Optional[torch.Tensor]] = []

        for _ in range(args.batch_size):
            s = samples[sample_index]
            sample_index = (sample_index + 1) % len(samples)

            batch_prompts.append(s.prompt)
            batch_images.append(str(s.image_path))

            gt_latent = None
            gt_video = None

            if args.dataset_use_latents and s.latent_path is not None:
                latent = _load_tensor(s.latent_path)
                gt_latent = latent.to(device)
            elif s.video_path is not None:
                gt_video = _load_single_video(s.video_path).to(device)

            batch_gt_videos.append(gt_video)
            batch_gt_latents.append(gt_latent)

        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.zeros([], device=device)
        total_video_mse = 0.0

        for prompt, image_path, gt_video, gt_latent in zip(
            batch_prompts, batch_images, batch_gt_videos, batch_gt_latents
        ):
            with torch.cuda.amp.autocast(
                enabled=bool(args.amp) and device.type == "cuda"
            ):
                outputs = guide.training_step(
                    prompt=prompt,
                    image=image_path,
                    gt_video=gt_video,
                    gt_points=None,
                    gt_latent=gt_latent,
                    video_weight=args.video_weight,
                    latent_weight=args.latent_weight,
                    point_weight=args.point_weight,
                    frame_num=frame_num,
                    offload_model=args.offload_model,
                )
                loss = outputs["loss"] / args.batch_size
            total_loss = total_loss + loss
            if "video" in outputs["losses"]:
                total_video_mse += (
                    outputs["losses"]["video"].detach().float().item()
                )

        scaler.scale(total_loss).backward()
        if args.max_grad_norm and args.max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                _optimizer_parameters(guide), args.max_grad_norm
            )
        scaler.step(optimizer)
        scaler.update()

        step += 1
        loss_scalar = total_loss.detach().float().item()
        running_loss += loss_scalar
        running_video_mse += total_video_mse
        running_steps += 1

        if step % args.log_every == 0 or step == args.steps:
            avg_loss = running_loss / max(running_steps, 1)
            avg_mse = running_video_mse / max(running_steps, 1e-8)
            psnr = _psnr(avg_mse) if running_video_mse > 0 else float("nan")
            logging.info(
                "Step %d/%d | loss=%.4f | video_mse=%.4f | psnr=%.2f dB",
                step,
                args.steps,
                avg_loss,
                avg_mse,
                psnr,
            )
            running_loss = 0.0
            running_video_mse = 0.0
            running_steps = 0

        if args.save_every and args.save_every > 0 and step % args.save_every == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = args.save_dir / f"wan_step_{step:06d}.pt"
            torch.save(guide.wan.model.state_dict(), ckpt_path)
            logging.info("Saved intermediate Wan checkpoint to %s", ckpt_path)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = args.save_dir / "wan_finetuned.pt"
    torch.save(guide.wan.model.state_dict(), final_ckpt)
    logging.info("Saved final Wan checkpoint to %s", final_ckpt)


def main():
    args = _parse_args()
    _setup_logger()
    logging.info("Starting Poze finetuning with args: %s", args)

    if args.dataset_root is not None:
        _train_dataset(args)
    else:
        _train_single_example(args)


if __name__ == "__main__":
    main()

