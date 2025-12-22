#!/usr/bin/env python3
"""
Minimal Wan TI2V finetuning loop guided by Pi3 latents.

This script mirrors the README training snippet by exposing a small CLI that
loads Wan and Pi3, computes video/point losses, and updates Wan parameters.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union


def _add_repo_to_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    for rel in ("wan", "pi3"):
        candidate = repo_root / rel
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    return repo_root


REPO_ROOT = _add_repo_to_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from wan import configs  # noqa: E402
from wan.pi3_guidance import Pi3GuidedTI2V  # noqa: E402
from wan.utils.utils import str2bool  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune Wan TI2V with Pi3-guided latents.")
    parser.add_argument(
        "--wan-ckpt-dir",
        required=True,
        help="Directory containing Wan2.2 TI2V checkpoints.")
    parser.add_argument(
        "--wan-config",
        default="ti2v-5B",
        choices=list(configs.WAN_CONFIGS.keys()),
        help="Wan TI2V config key to load.")
    parser.add_argument(
        "--pi3-checkpoint",
        default=None,
        help="Optional local Pi3 checkpoint path; if omitted, pulls the pretrained id."
    )
    parser.add_argument(
        "--pi3-pretrained-id",
        default="yyfz233/Pi3",
        help="HuggingFace model id for Pi3 when no checkpoint is provided.")
    parser.add_argument(
        "--use-pi3",
        type=str2bool,
        default=True,
        help="Enable Pi3-guided conditioning; disable to finetune Wan without Pi3 inputs."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to condition Wan generation.")
    parser.add_argument(
        "--image",
        required=True,
        help="Reference image path used for TI2V conditioning.")
    parser.add_argument(
        "--gt-video",
        default=None,
        help="Path to a torch tensor (.pt/.pth) ground-truth video shaped like the Wan output (C, F, H, W)."
    )
    parser.add_argument(
        "--gt-points",
        default=None,
        help="Path to a torch tensor (.pt/.pth) point cloud shaped like Pi3 outputs (B, N, H, W, 3)."
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=None,
        help="Number of frames to generate. Defaults to config.frame_num.")
    parser.add_argument(
        "--video-weight",
        type=float,
        default=1.0,
        help="Weight for video reconstruction loss.")
    parser.add_argument(
        "--point-weight",
        type=float,
        default=1.0,
        help="Weight for point cloud reconstruction loss.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for Wan finetuning.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of optimization steps to run.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="How often (in steps) to log losses.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save a Wan checkpoint every N steps (0 disables interim checkpoints).")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "finetune_outputs",
        help="Directory to store checkpoints.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to Pi3GuidedTI2V, e.g., cuda or cpu.")
    parser.add_argument(
        "--offload-model",
        type=str2bool,
        default=False,
        help="Offload Wan to CPU between steps. Keep False for finetuning.")
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Use torch.cuda.amp for mixed-precision training when available.")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional gradient clipping norm.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.")
    return parser.parse_args()


def _load_tensor(path: Optional[str], device: Union[torch.device, str]):
    """
    Load a tensor from disk.

    Args:
        path: Path to a serialized tensor file. If None, returns None.
        device: Device mapping passed to torch.load.

    Returns:
        Loaded torch.Tensor or None when path is None.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded object is not a tensor.
    """
    if path is None:
        return None
    tensor_path = Path(path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    tensor = torch.load(tensor_path, map_location=device)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Loaded object from {tensor_path} is not a tensor.")
    return tensor


def _validate_targets(gt_video, gt_points):
    """
    Ensure at least one supervision signal is provided.

    Raises:
        ValueError: If both gt_video and gt_points are None.
    """
    if gt_video is None and gt_points is None:
        raise ValueError("At least one of --gt-video or --gt-points must be provided.")


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cfg = configs.WAN_CONFIGS[args.wan_config]
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    target_device = torch.device(args.device)

    guide = Pi3GuidedTI2V(
        wan_config=cfg,
        wan_checkpoint_dir=args.wan_ckpt_dir,
        pi3_checkpoint=args.pi3_checkpoint,
        pi3_pretrained_id=args.pi3_pretrained_id,
        device=target_device,
        trainable_wan=True,
        use_pi3=args.use_pi3,
    )
    guide.wan.model.train()

    params = [p for p in guide.wan.model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "No Wan parameters are marked trainable. Ensure trainable_wan=True or unfreeze layers."
        )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    gt_video = _load_tensor(args.gt_video, target_device)
    gt_points = _load_tensor(args.gt_points, target_device)
    _validate_targets(gt_video, gt_points)

    pil_image = Image.open(args.image).convert("RGB")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = guide.training_step(
                prompt=args.prompt,
                image=pil_image,
                gt_video=gt_video,
                gt_points=gt_points,
                video_weight=args.video_weight,
                point_weight=args.point_weight,
                frame_num=frame_num,
                offload_model=args.offload_model,
            )
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            loss_items = {k: v.item() for k, v in outputs.get("losses", {}).items()}
            logging.info(
                "Step %d/%d - total_loss=%.6f details=%s",
                step,
                args.steps,
                float(loss.detach().cpu()),
                loss_items,
            )

        should_save = args.save_every and step % args.save_every == 0
        if should_save or step == args.steps:
            ckpt_path = args.save_dir / f"wan_finetune_step{step}.pt"
            logging.info("Saving Wan checkpoint to %s", ckpt_path)
            torch.save(guide.wan.model.state_dict(), ckpt_path)

    logging.info("Finetuning finished.")


if __name__ == "__main__":
    main()
