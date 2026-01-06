import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    @staticmethod
    def _ensure_view_dim(frame_first_latent: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Guarantee an explicit view dimension (V) for frame-first Pi3 latents.
        Converts (F, HW, C) to (F, 1, HW, C) when the view axis is missing.
        """
        if frame_first_latent is not None and frame_first_latent.dim() == 3:
            return frame_first_latent.unsqueeze(1)
        return frame_first_latent

    @staticmethod
    def _nested_frames_list(tensor: torch.Tensor):
        """Convert (B, F, ...) tensor into nested [batch][frame] list for per-frame access."""
        return [list(torch.unbind(batch_tensor, dim=0)) for batch_tensor in torch.unbind(tensor, dim=0)]

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

    def build_latent_volume(
        self,
        latents: Dict[str, Any],
        *,
        default_patch_size: Optional[int] = None,
        default_patch_start_idx: Optional[int] = None,
    ) -> torch.Tensor:
        patch_size = latents.get("patch_size", default_patch_size or getattr(self.pi3, "patch_size", None))
        patch_start_idx = latents.get("patch_start_idx", default_patch_start_idx or getattr(self.pi3, "patch_start_idx", None))
        if patch_size is None:
            raise ValueError(
                "patch_size must be provided to build Pi3 latent volume (set via configure_pi3_adapters or latents['patch_size'])."
            )
        if patch_start_idx is None:
            raise ValueError(
                "patch_start_idx must be provided to build Pi3 latent volume (set via configure_pi3_adapters or latents['patch_start_idx'])."
            )

        patch_h = latents["hw"][0] // patch_size
        patch_w = latents["hw"][1] // patch_size
        tokens = latents["hidden"][:, patch_start_idx:, :]
        tokens = tokens.view(
            latents["batch"],
            latents["frames"],
            patch_h,
            patch_w,
            tokens.size(-1),
        )
        return tokens.permute(0, 4, 1, 2, 3).contiguous()

    def prepare_condition(
        self,
        video_condition: Any,
        cond_latent: torch.Tensor,
        *,
        device: torch.device,
        latent_adapter: Optional[torch.nn.Module],
        use_pi3_condition: bool,
        concat_method: str,
        default_patch_size: Optional[int] = None,
        default_patch_start_idx: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], int]:
        if video_condition is None or not use_pi3_condition:
            return None, 0

        cond = video_condition
        if isinstance(cond, dict):
            cond = self.build_latent_volume(
                cond,
                default_patch_size=default_patch_size,
                default_patch_start_idx=default_patch_start_idx,
            )
        if cond is None:
            return None, 0
        if isinstance(cond, list):
            if len(cond) == 0:
                return None, 0
            cond = cond[0]
        if cond.dim() == 4:
            cond = cond.unsqueeze(0)
        if cond.dim() != 5:
            raise ValueError("video_condition must have shape (C, F, H, W) or (B, C, F, H, W).")
        cond = cond.to(device=device, dtype=cond_latent.dtype)

        target_size = (cond_latent.shape[1], cond_latent.shape[2], cond_latent.shape[3])
        if cond.shape[-3:] != target_size:
            cond = torch.nn.functional.interpolate(cond, size=target_size, mode="trilinear", align_corners=False)

        conv_output = latent_adapter(cond) if latent_adapter is not None else cond
        conv_output = conv_output.contiguous()

        logger.debug("Prepared Pi3 condition with shape %s", conv_output.shape)
        return conv_output.squeeze(0), conv_output.shape[1]

    def recover_latents(
        self,
        pi3_latent: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        target_size: Tuple[int, int, int],
        *,
        recover_adapter: Optional[torch.nn.Module] = None,
        flatten_to_frames: bool = False,
    ) -> Optional[torch.Tensor]:
        if pi3_latent is None:
            return None
        if isinstance(pi3_latent, list):
            if len(pi3_latent) == 0:
                return None
            pi3_latent = pi3_latent[0]

        logger.debug("Pi3 latent before recovery shape: %s", getattr(pi3_latent, "shape", None))

        if recover_adapter is None:
            return pi3_latent

        processed_latent = pi3_latent
        if processed_latent.dim() == 4:
            processed_latent = processed_latent.unsqueeze(0)
        if processed_latent.dim() != 5:
            return None

        try:
            target_device = next(recover_adapter.parameters()).device
        except StopIteration:
            target_device = processed_latent.device
        processed_latent = processed_latent.to(target_device)

        if processed_latent.shape[-3:] != target_size:
            processed_latent = F.interpolate(
                processed_latent,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        if processed_latent.shape[1] != recover_adapter.in_channels:
            in_ch = processed_latent.shape[1]
            exp_ch = recover_adapter.in_channels
            if exp_ch % in_ch == 0:
                repeat_factor = exp_ch // in_ch
                processed_latent = processed_latent.repeat(1, repeat_factor, 1, 1, 1)
            elif in_ch > exp_ch:
                processed_latent = processed_latent[:, :exp_ch]
            else:
                pad_ch = exp_ch - in_ch
                pad = torch.zeros(
                    (processed_latent.shape[0], pad_ch, *processed_latent.shape[2:]),
                    device=processed_latent.device,
                    dtype=processed_latent.dtype,
                )
                processed_latent = torch.cat([processed_latent, pad], dim=1)

        recovered = recover_adapter(processed_latent)

        if flatten_to_frames:
            batch_dim = recovered.shape[0]
            channel_dim = recovered.shape[1]
            frame_count = recovered.shape[2]
            spatial_h, spatial_w = recovered.shape[3], recovered.shape[4]
            recovered = recovered.permute(2, 0, 3, 4, 1).contiguous().view(
                frame_count,
                batch_dim,
                spatial_h * spatial_w,
                channel_dim,
            )
            return recovered.unsqueeze(1) if recovered.dim() == 3 else recovered

        return recovered.squeeze(0) if recovered.shape[0] == 1 else recovered

    def decode_latent_sequence(self, pi3_latent: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
        if self.pi3 is None or pi3_latent is None:
            return None
        with torch.no_grad():
            recovered = self._ensure_view_dim(pi3_latent)
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
                points_list = self._nested_frames_list(points)
                if has_conf:
                    conf_list = self._nested_frames_list(conf)
                    decoded["conf_list"] = conf_list
                decoded["points_list"] = points_list
            logger.debug("Decoded Pi3 latents into shapes: %s", {k: v.shape if torch.is_tensor(v) else type(v) for k, v in decoded.items()})
            return decoded

    def forward(
        self,
        pi3_latent: Optional[torch.Tensor],
        *,
        latents_meta: Optional[Dict[str, Any]] = None,
        cache_metadata: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if cache_metadata:
            self.cache_latent_metadata(latents_meta)
        return self.decode_latent_sequence(pi3_latent)
