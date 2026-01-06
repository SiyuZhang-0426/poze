import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pi3.utils.basic import load_images_as_tensor

logger = logging.getLogger(__name__)

EXPECTED_POINT_TENSOR_DIMS = 5
# Pi3 decoder expects views; keep per-call decode to a single frame to avoid mixing frames as views.
FRAMES_PER_DECODE = 1
MIN_RESHAPE_DIMS = 3


def ensure_view_dim(frame_first_latent: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Guarantee an explicit view dimension (V) for frame-first Pi3 latents.
    Converts (F, HW, C) to (F, 1, HW, C) when the view axis is missing.
    """
    if frame_first_latent is not None and frame_first_latent.dim() == 3:
        return frame_first_latent.unsqueeze(1)
    return frame_first_latent


def _nested_frames_list(tensor: torch.Tensor):
    """Convert (B, F, ...) tensor into nested [batch][frame] list for per-frame access."""
    return [list(torch.unbind(batch_tensor, dim=0)) for batch_tensor in torch.unbind(tensor, dim=0)]


class Pi3StitchingLayer(nn.Module):
    """
    Handle padding and tensor preparation so images align with Pi3 patch sizing.
    """

    def __init__(self, pi3_model, device: Union[torch.device, str]) -> None:
        super().__init__()
        self.pi3 = pi3_model
        self.device = torch.device(device)

    def _patch_size(self) -> int:
        if self.pi3 is None:
            raise ValueError("Pi3 model is required to determine patch size.")
        return self.pi3.patch_size

    def ensure_divisible_size(self, image):
        """
        Pad a PIL image so height and width are divisible by the Pi3 patch size.
        """
        patch = self._patch_size()
        target_w = math.ceil(image.width / patch) * patch
        target_h = math.ceil(image.height / patch) * patch
        if target_w == image.width and target_h == image.height:
            return image
        padding = (0, 0, target_w - image.width, target_h - image.height)
        return TF.pad(image, padding, fill=0)

    def pad_tensor_divisible(self, tensor: torch.Tensor) -> torch.Tensor:
        patch = self._patch_size()
        leading_shape = tensor.shape[:-3]
        c, h, w = tensor.shape[-3:]
        target_h = math.ceil(h / patch) * patch
        target_w = math.ceil(w / patch) * patch
        if target_h == h and target_w == w:
            return tensor
        padded = F.pad(tensor.reshape(-1, c, h, w), (0, target_w - w, 0, target_h - h))
        return padded.reshape(*leading_shape, c, target_h, target_w)

    def _normalize_pil(self, image):
        return image if hasattr(image, "mode") and image.mode == "RGB" else image.convert("RGB")

    def prepare_image_inputs(self, image, use_pi3: bool) -> Tuple[torch.Tensor, Any]:
        """
        Normalize various image inputs into tensor+PIL pairs suited for Wan/Pi3.
        """
        if isinstance(image, (str, Path)):
            tensor = load_images_as_tensor(str(image), interval=1)
            if tensor.numel() == 0:
                raise ValueError(
                    f"No image loaded from {image}; ensure the path points to a readable .png/.jpg/.jpeg file or directory."
                )
            image = tensor
        if not use_pi3 or self.pi3 is None:
            if isinstance(image, torch.Tensor):
                base = image
                if base.dim() == 4:
                    base = base[0]
                if base.dim() != 3:
                    raise ValueError(
                        f"Expected image tensor with shape (C, H, W) or (N, C, H, W); got {tuple(image.shape)}"
                    )
                base_cpu = base if base.device.type == "cpu" else base.cpu()
                pil = TF.to_pil_image(base_cpu)
                tensor = base.unsqueeze(0).unsqueeze(0)
            else:
                pil = self._normalize_pil(image)
                tensor = TF.to_tensor(pil).unsqueeze(0).unsqueeze(0)
            return tensor.to(self.device), pil

        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            needs_pad = tensor.shape[-1] % self._patch_size() != 0 or tensor.shape[-2] % self._patch_size() != 0
            if needs_pad:
                tensor = self.pad_tensor_divisible(tensor)
            pil = TF.to_pil_image(tensor[0, 0].cpu())
        else:
            pil = self.ensure_divisible_size(self._normalize_pil(image))
            tensor = TF.to_tensor(pil).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), pil

    def forward(self, image, use_pi3: bool):
        return self.prepare_image_inputs(image, use_pi3)


class Pi3RecoverLayer(nn.Module):
    """
    Decode dynamic Pi3 latents back to Pi3 outputs with cached token shape metadata.
    """

    def __init__(self, pi3_model) -> None:
        super().__init__()
        self.pi3 = pi3_model
        self._pi3_shape: Optional[Tuple[int, int, int]] = None
        self._pi3_hw: Optional[Tuple[int, int]] = None

    def cache_latent_metadata(self, latents: Optional[Dict[str, Any]]) -> None:
        self._pi3_shape = None
        self._pi3_hw = None
        if latents is None:
            return
        hidden = latents.get("hidden")
        if hidden is not None:
            self._pi3_shape = hidden.shape
        self._pi3_hw = latents.get("hw")

    def _expected_hw_tokens(self) -> Optional[int]:
        if self._pi3_shape is None or self.pi3 is None:
            return None
        return max(1, self._pi3_shape[1] - self.pi3.patch_start_idx)

    def decode_latent_sequence(self, pi3_latent: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
        if self.pi3 is None or pi3_latent is None:
            return None
        with torch.no_grad():
            recovered = ensure_view_dim(pi3_latent)
            expected_hw = self._expected_hw_tokens()
            frame_first = (
                recovered is not None
                and recovered.dim() == 4
                and recovered.shape[-1] == 2 * self.pi3.dec_embed_dim
                and (expected_hw is None or recovered.shape[-2] == expected_hw)
            )
            if frame_first:
                f, b, hw, c = recovered.shape
                batch_first = recovered.permute(1, 0, 2, 3)  # (B, F, HW, C)
                b, f, hw, c = batch_first.shape
                patch_size = self.pi3.patch_size
                if self._pi3_hw is not None:
                    input_h_pix, input_w_pix = self._pi3_hw
                    h = max(1, input_h_pix // patch_size)
                    w = max(1, input_w_pix // patch_size)
                else:
                    root = int(math.sqrt(hw))
                    h = next((cand for cand in range(root, 0, -1) if hw % cand == 0), 1)
                    w = hw // h
                tokens = batch_first.reshape(b, f, hw, c)
            else:
                if recovered.dim() == 4:
                    recovered = recovered.unsqueeze(0)
                b, c, f, h, w = recovered.shape
                tokens = recovered.permute(0, 2, 3, 4, 1).reshape(b, f, h * w, c)

            register = torch.cat(
                [self.pi3.register_token, self.pi3.register_token],
                dim=-1,
            ).to(tokens.device, tokens.dtype)
            register = register.repeat(b, f, 1, 1).reshape(
                b,
                f,
                self.pi3.patch_start_idx,
                tokens.shape[-1],
            )
            embed_dim = tokens.shape[3]
            pos = self.pi3.position_getter(b, h, w, tokens.device)
            pos = pos + 1
            pos_special = torch.zeros(
                b,
                self.pi3.patch_start_idx,
                2,
                device=tokens.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1).repeat_interleave(f, dim=0)

            H_pix = h * self.pi3.patch_size
            W_pix = w * self.pi3.patch_size
            register_flat = register.reshape(b * f, self.pi3.patch_start_idx, embed_dim)
            tokens_flat = tokens.reshape(b * f, tokens.shape[2], embed_dim)
            decoder_hidden_flat = torch.cat([register_flat, tokens_flat], dim=1)
            decoded_raw = self.pi3._decode_tokens(
                decoder_hidden_flat,
                pos,
                H_pix,
                W_pix,
                b * f,
                FRAMES_PER_DECODE,
            )

            def _reshape_decoded_value(val: torch.Tensor) -> torch.Tensor:
                if val.shape[0] == b * f and val.dim() >= MIN_RESHAPE_DIMS:
                    return val.reshape(b, f, *val.shape[2:])
                return val

            decoded = {key: _reshape_decoded_value(value) for key, value in decoded_raw.items()}
            points = decoded.get("points")
            conf = decoded.get("conf")
            has_conf = conf is not None and conf.dim() == EXPECTED_POINT_TENSOR_DIMS
            if points is not None and points.dim() == EXPECTED_POINT_TENSOR_DIMS:
                points_list = _nested_frames_list(points)
                if has_conf:
                    conf_list = _nested_frames_list(conf)
                    decoded["conf_list"] = conf_list
                decoded["points_list"] = points_list
            logger.debug("Decoded Pi3 latents into shapes: %s", {k: v.shape if torch.is_tensor(v) else type(v) for k, v in decoded.items()})
            return decoded

    def forward(self, pi3_latent: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
        return self.decode_latent_sequence(pi3_latent)
