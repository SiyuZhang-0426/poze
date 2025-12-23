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
from typing import Any, Dict, List, Optional, Union


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
from safetensors.torch import load_file  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from wan import configs  # noqa: E402
from wan.pi3_guidance import Pi3GuidedTI2V  # noqa: E402
from wan.utils.utils import str2bool  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune Wan TI2V with Pi3-guided latents."
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
        help="Enable Pi3-guided conditioning; disable to finetune Wan without Pi3 inputs."
    )
    parser.add_argument(
        "--concat-method",
        default="channel",
        choices=("channel", "frame", "width", "height"),
        help="How to fuse Pi3 latents with RGB latents: channel (default), frame, width, or height.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt to condition Wan generation."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Reference image path used for TI2V conditioning."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional 4DNeX-style dataset root. When set, prompts/images/videos are drawn from this directory instead of single-sample inputs.",
    )
    parser.add_argument(
        "--dataset-prompt-file",
        default="prompts.txt",
        help="Relative prompt file inside dataset root."
    )
    parser.add_argument(
        "--dataset-video-list",
        default="videos.txt",
        help="Relative video list file inside dataset root."
    )
    parser.add_argument(
        "--dataset-image-dir",
        default="first_frames",
        help="Relative directory of first-frame images inside dataset root."
    )
    parser.add_argument(
        "--dataset-latent-dir",
        type=Path,
        default=None,
        help="Optional explicit path to Wan VAE latents (.safetensors). Defaults to the first cache directory under dataset root.",
    )
    parser.add_argument(
        "--dataset-use-latents",
        type=str2bool,
        default=True,
        help="Prefer Wan VAE latents from the dataset cache for reconstruction loss when available.",
    )
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
        help="Number of frames to generate. Defaults to config.frame_num."
    )
    parser.add_argument(
        "--video-weight",
        type=float,
        default=1.0,
        help="Weight for video reconstruction loss."
    )
    parser.add_argument(
        "--latent-weight",
        type=float,
        default=1.0,
        help="Weight for latent reconstruction loss when cached Wan latents are available."
    )
    parser.add_argument(
        "--point-weight",
        type=float,
        default=1.0,
        help="Weight for point cloud reconstruction loss."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for Wan finetuning."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of optimization steps to run."
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="How often (in steps) to log losses."
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save a Wan checkpoint every N steps (0 disables interim checkpoints)."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "finetune_outputs",
        help="Directory to store checkpoints."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dataset finetuning. Only batch size 1 is supported."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the dataset DataLoader."
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling during finetuning."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to Pi3GuidedTI2V, e.g., cuda or cpu."
    )
    parser.add_argument(
        "--offload-model",
        type=str2bool,
        default=False,
        help="Offload Wan to CPU between steps. Keep False for finetuning."
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Use torch.cuda.amp for mixed-precision training when available."
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional gradient clipping norm."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility."
    )
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


def _validate_targets(gt_video, gt_points, gt_latent=None):
    """
    Ensure at least one supervision signal is provided.

    Raises:
        ValueError: If all supervision targets are None.
    """
    if gt_video is None and gt_points is None and gt_latent is None:
        raise ValueError("At least one of --gt-video, --gt-points, or cached latents must be provided.")


class FourDNeXDataset(Dataset):
    """
    Lightweight 4DNeX loader that mirrors the preprocessed layout produced by
    `4DNeX/build_wan_dataset.py`.

    Expected structure under ``root``:
    - prompts.txt: one prompt per line
    - videos.txt: one relative video path per line (e.g., ``videos/xxx.mp4``)
    - first_frames/: optional PNGs matching video stems
    - cache/video_latent/wan-i2v/<resolution>/: optional Wan VAE latents (*.safetensors)
    """

    def __init__(
        self,
        root: Union[str, Path],
        prompt_file: str = "prompts.txt",
        video_list: str = "videos.txt",
        image_dir: str = "first_frames",
        latent_dir: Optional[Union[str, Path]] = None,
        use_latents: bool = True,
        prompt_suffix: str = "POINTMAP_STYLE.",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.prompts = self._read_lines(self.root / prompt_file)
        self.videos = [self.root / v for v in self._read_lines(self.root / video_list)]
        self.image_dir = self.root / image_dir
        self.latent_dir = self._resolve_latent_dir(latent_dir)
        self.use_latents = use_latents
        self.prompt_suffix = prompt_suffix.strip()

        if len(self.prompts) != len(self.videos):
            raise ValueError(
                f"Dataset file count mismatch: {len(self.prompts)} prompts vs {len(self.videos)} videos."
            )
        if not self.prompts:
            raise ValueError("No samples found in dataset.")

    def _read_lines(self, path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _resolve_latent_dir(self, latent_dir: Optional[Union[str, Path]]) -> Optional[Path]:
        if latent_dir is not None:
            latent_path = Path(latent_dir)
            return latent_path if latent_path.exists() else None
        candidate = self.root / "cache" / "video_latent" / "wan-i2v"
        if candidate.exists():
            subdirs = sorted([p for p in candidate.iterdir() if p.is_dir()])
            if subdirs:
                return subdirs[0]
        return None

    def __len__(self) -> int:
        return len(self.videos)

    def _load_latent(self, video_name: str) -> Optional[torch.Tensor]:
        if self.latent_dir is None:
            return None
        latent_path = self.latent_dir / f"{video_name}.safetensors"
        if not latent_path.exists():
            return None
        loaded = load_file(latent_path)
        latent = loaded.get("encoded_video")
        if latent is None:
            return None
        return latent.float()

    def _decode_video(self, video_path: Path) -> torch.Tensor:
        frames = None
        # decord/torchvision are optional; import lazily to avoid hard dependencies when the dataset loader is unused.
        try:
            import decord

            decord.bridge.set_bridge("torch")
            reader = decord.VideoReader(uri=str(video_path))
            frames = reader.get_batch(range(len(reader)))  # (F, H, W, 3)
            if not torch.is_floating_point(frames):
                frames = frames.float()
            frames = frames.permute(3, 0, 1, 2) / 255.0
        except ImportError:
            frames = None
        except Exception as err:
            logging.warning("Decord failed to read %s (%s); falling back to torchvision.read_video", video_path, err)
            frames = None
        if frames is None:
            from torchvision.io import read_video

            frames, _, _ = read_video(str(video_path), output_format="TCHW")
            frames = frames.permute(1, 0, 2, 3).float() / 255.0  # (C, F, H, W)
        return frames * 2.0 - 1.0  # (C, F, H, W) in [-1, 1]

    def _load_image(self, video_name: str, video: Optional[torch.Tensor]) -> Image.Image:
        first_frame = self.image_dir / f"{video_name}.png"
        if first_frame.exists():
            return Image.open(first_frame).convert("RGB")
        if video is not None:
            # Keep this import local to avoid requiring torchvision when only latent supervision is used.
            from torchvision.transforms.functional import to_pil_image

            frame = ((video[:, 0] + 1) * 0.5).clamp(0, 1)
            return to_pil_image(frame.cpu())
        raise FileNotFoundError(f"First frame not found for {video_name} and no video available to derive it.")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        prompt = self.prompts[index]
        if self.prompt_suffix:
            prompt = f"{prompt} {self.prompt_suffix}".strip()
        video_path = self.videos[index]
        video_name = video_path.stem

        video_latent = self._load_latent(video_name) if self.use_latents else None
        video_tensor = None
        if (not self.use_latents) or video_latent is None:
            video_tensor = self._decode_video(video_path)

        frame_num = None
        if video_latent is not None:
            frame_num = video_latent.shape[1]
        elif video_tensor is not None:
            frame_num = video_tensor.shape[1]

        image = self._load_image(video_name, video_tensor)

        if video_latent is None and video_tensor is None:
            raise ValueError(f"No supervision available for sample {video_name}")

        return {
            "prompt": prompt,
            "image": image,
            "video": video_tensor,
            "video_latent": video_latent,
            "frame_num": frame_num,
            "video_path": video_path,
        }


def _single_sample_collate(batch):
    """
    Pi3-guided Wan finetuning currently supports batch size 1; unwrap the single element.
    """
    return batch[0]


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

    if args.dataset_root is None:
        if args.prompt is None or args.image is None:
            raise ValueError("Provide --prompt and --image or set --dataset-root for dataset finetuning.")
    else:
        if args.batch_size != 1:
            raise ValueError("Only batch size 1 is supported in the current finetuning loop because Pi3-guided inputs are single-image.")

    cfg = configs.WAN_CONFIGS[args.wan_config]
    default_frame_num = args.frame_num if args.frame_num is not None else cfg.frame_num
    target_device = torch.device(args.device)

    dataset = None
    dataloader = None
    data_iter = None
    if args.dataset_root is not None:
        dataset = FourDNeXDataset(
            root=args.dataset_root,
            prompt_file=args.dataset_prompt_file,
            video_list=args.dataset_video_list,
            image_dir=args.dataset_image_dir,
            latent_dir=args.dataset_latent_dir,
            use_latents=args.dataset_use_latents,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            num_workers=args.num_workers,
            collate_fn=_single_sample_collate,
        )
        data_iter = iter(dataloader)

    guide = Pi3GuidedTI2V(
        wan_config=cfg,
        wan_checkpoint_dir=args.wan_ckpt_dir,
        pi3_checkpoint=args.pi3_checkpoint,
        pi3_pretrained_id=args.pi3_pretrained_id,
        device=target_device,
        trainable_wan=True,
        use_pi3=args.use_pi3,
        concat_method=args.concat_method,
    )
    if getattr(guide, "pi3", None) is not None:
        guide.pi3.eval().requires_grad_(False)
    guide.wan.model.train()
    for extra in (getattr(guide, "latent_adapter", None), getattr(guide, "pi3_recover_adapter", None)):
        if extra is not None:
            extra.train()

    params = [p for p in guide.wan.model.parameters() if p.requires_grad]
    for extra in (getattr(guide, "latent_adapter", None), getattr(guide, "pi3_recover_adapter", None)):
        if extra is not None:
            params.extend(p for p in extra.parameters() if p.requires_grad)
    if not params:
        raise RuntimeError(
            "No Wan parameters are marked trainable. Ensure trainable_wan=True or unfreeze layers."
        )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    gt_video = None
    gt_points = None
    gt_latent = None
    pil_image = None
    if dataset is None:
        gt_video = _load_tensor(args.gt_video, target_device)
        gt_points = _load_tensor(args.gt_points, target_device)
        _validate_targets(gt_video, gt_points)
        pil_image = Image.open(args.image).convert("RGB")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        sample = None
        if dataset is not None:
            try:
                sample = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                sample = next(data_iter)
            prompt = sample["prompt"]
            pil_image = sample["image"]
            gt_video = sample.get("video")
            gt_latent = sample.get("video_latent")
            gt_points = None
            frame_num = sample.get("frame_num") or default_frame_num
            if gt_video is not None:
                gt_video = gt_video.to(target_device)
            if gt_latent is not None:
                gt_latent = gt_latent.to(target_device)
            _validate_targets(gt_video, gt_points, gt_latent)
        else:
            prompt = args.prompt
            frame_num = default_frame_num

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = guide.training_step(
                prompt=prompt,
                image=pil_image,
                gt_video=gt_video,
                gt_points=gt_points,
                gt_latent=gt_latent,
                video_weight=args.video_weight,
                latent_weight=args.latent_weight,
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
