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

from pi3.utils.basic import write_ply
from wan import configs
from wan.pi3_guidance import Pi3GuidedTI2V
from wan.utils.utils import save_video, str2bool


PI3_CONF_THRESHOLD = 0.5


def save_pi3(
    outputs: dict,
    save_path: Path,
    image_path: str
) -> None:
    pi3_out = outputs.get("pi3_preds")
    if pi3_out is None:
        logging.warning("Pi3 predictions unavailable; skipping save_pi3.")
        return
    points = pi3_out.get("points")
    if points is None:
        logging.warning("Pi3 predictions missing points; skipped PLY export.")
        return
    conf = pi3_out.get("conf")
    save_dir = save_path
    if save_dir.suffix.lower() == ".ply":
        save_dir = save_dir.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(image_path).stem or "frame"
    points_tensor = points
    conf_tensor = conf
    if points_tensor.dim() == 4:
        points_tensor = points_tensor.unsqueeze(0)
    if conf_tensor is not None and conf_tensor.dim() == 4:
        conf_tensor = conf_tensor.unsqueeze(0)
    if points_tensor.dim() != 5:
        logging.warning(
            "Unexpected Pi3 points shape %s; expected (B, F, H, W, 3). Skipped PLY export.",
            tuple(points_tensor.shape),
        )
        return
    batch_size, frame_count = points_tensor.shape[:2]
    for batch_idx in range(batch_size):
        batch_prefix = base_name if batch_size == 1 else f"{base_name}_b{batch_idx}"
        for frame_idx in range(frame_count):
            frame_points = points_tensor[batch_idx, frame_idx]
            frame_conf = None
            if conf_tensor is not None and conf_tensor.dim() >= 4:
                if conf_tensor.shape[0] == batch_size and conf_tensor.shape[1] == frame_count:
                    frame_conf = conf_tensor[batch_idx, frame_idx]
            points_to_save = frame_points
            if frame_conf is not None:
                conf_map = frame_conf.squeeze(-1)
                expected_mask_shape = points_to_save.shape[:-1]
                if conf_map.shape[: len(expected_mask_shape)] == expected_mask_shape:
                    conf_mask = conf_map > PI3_CONF_THRESHOLD
                    if conf_mask.any():
                        points_to_save = points_to_save[conf_mask]
                else:
                    logging.warning(
                        "Pi3 confidence shape %s does not match points shape %s; skipping confidence filtering for frame %d.",
                        tuple(conf_map.shape),
                        tuple(expected_mask_shape),
                        frame_idx,
                    )
            ply_name = f"{batch_prefix}_{frame_idx:04d}.ply"
            ply_path = save_dir / ply_name
            write_ply(points_to_save, path=str(ply_path))
    logging.info(
        "Saved %d Pi3 point cloud frame(s) to %s",
        batch_size * frame_count,
        save_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pi3-guided Wan TI2V inference."
    )
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
        "--save-pi3",
        type=Path,
        default=None,
        help="Optional directory (or .ply path for backward compatibility) to save per-frame Pi3 point clouds."
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
    outputs = guide.generate_with_3d(
        prompt=args.prompt,
        image=args.image,
        frame_num=frame_num,
        offload_model=args.offload_model,
        enable_grad=False,
        decode_pi3=bool(args.save_pi3),
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

    if args.save_pi3:
        save_pi3(outputs=outputs, save_path=args.save_pi3, image_path=args.image)

    logging.info("Done.")


if __name__ == "__main__":
    main()
