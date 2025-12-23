#!/usr/bin/env python3
"""
Pi3-guided Wan TI2V inference script.

This wraps the README example into a runnable CLI that loads Pi3 and Wan,
generates a video guided by Pi3 3D latents, and optionally saves latents.
"""
import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path


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
from wan.utils.utils import save_video, str2bool  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pi3-guided Wan TI2V inference.")
    parser.add_argument(
        "--wan-ckpt-dir",
        required=True,
        help="Directory containing Wan2.2 TI2V checkpoints."
    )
    parser.add_argument(
        "--wan-config",
        default="ti2v-5B",
        choices=list(configs.WAN_CONFIGS.keys()),
        help="Wan TI2V config key to load."
    )
    parser.add_argument(
        "--pi3-checkpoint",
        default=None,
        help="Optional local Pi3 checkpoint path; if omitted, pulls the pretrained id."
    )
    parser.add_argument(
        "--pi3-pretrained-id",
        default="yyfz233/Pi3",
        help="HuggingFace model id for Pi3 when no checkpoint is provided."
    )
    parser.add_argument(
        "--use-pi3",
        type=str2bool,
        default=True,
        help="Enable Pi3-guided conditioning; set false to run Wan TI2V without Pi3."
    )
    parser.add_argument(
        "--concat-method",
        default="channel",
        choices=("channel", "frame", "width", "height"),
        help="How to fuse Pi3 latents with RGB latents: channel (default), frame, width, or height.",
    )
    parser.add_argument(
        "--prompt",
        default="A drone flythrough of a canyon",
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Reference image path used for TI2V conditioning."
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=None,
        help="Number of frames to generate. Defaults to config.frame_num."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to Pi3GuidedTI2V, e.g., cuda or cpu."
    )
    parser.add_argument(
        "--offload-model",
        type=str2bool,
        default=True,
        help="Offload Wan to CPU between steps to save VRAM (recommended for inference)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the generated video (mp4)."
    )
    parser.add_argument(
        "--save-latents",
        type=Path,
        default=None,
        help="Optional path to torch.save the returned latents dict."
    )
    parser.add_argument(
        "--save-pi3",
        type=Path,
        default=None,
        help="Optional path to torch.save the Pi3 decode outputs (points/depth/rgb)."
    )
    return parser.parse_args()


def _default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_stub = re.sub(r"[^\w\-]+", "_", args.prompt).strip("_")
    prompt_stub = prompt_stub[:50] or "prompt"
    return (REPO_ROOT / "outputs" /
            f"{args.wan_config}_{prompt_stub}_{timestamp}.mp4")


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = configs.WAN_CONFIGS[args.wan_config]
    frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    output_path = args.output or _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading Pi3-guided Wan TI2V pipeline...")
    guide = Pi3GuidedTI2V(
        wan_config=cfg,
        wan_checkpoint_dir=args.wan_ckpt_dir,
        pi3_checkpoint=args.pi3_checkpoint,
        pi3_pretrained_id=args.pi3_pretrained_id,
        device=args.device,
        trainable_wan=False,
        use_pi3=args.use_pi3,
        concat_method=args.concat_method,
    )

    logging.info("Running generation...")
    pil_image = Image.open(args.image).convert("RGB")
    outputs = guide.generate_with_3d(
        prompt=args.prompt,
        image=pil_image,
        frame_num=frame_num,
        offload_model=args.offload_model,
        enable_grad=False,
    )

    logging.info("Saving video to %s", output_path)
    video = outputs["video"].unsqueeze(0)
    save_video(
        tensor=video,
        save_file=str(output_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
