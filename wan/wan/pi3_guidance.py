import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from .textimage2video import WanTI2V


def ensure_view_dim(frame_first_latent: torch.Tensor | None) -> torch.Tensor | None:
    """
    Guarantee an explicit view dimension (V) for frame-first Pi3 latents.
    Converts (F, HW, C) to (F, 1, HW, C) when the view axis is missing.
    """
    if frame_first_latent is not None and frame_first_latent.dim() == 3:
        return frame_first_latent.unsqueeze(1)
    return frame_first_latent


DEFAULT_FRAME_NUM = 81


class Pi3GuidedTI2V(nn.Module):
    """
    Pipeline that freezes Pi3, adapts its latents, and optionally finetunes Wan2.2 TI2V.
    """

    def __init__(
        self,
        wan_config,
        wan_checkpoint_dir: str,
        pi3_checkpoint: Optional[str] = None,
        use_pi3: bool = True,
        device: str = "cuda",
        trainable_wan: bool = False,
        pi3_pretrained_id: str = "yyfz233/Pi3",
        pi3_weights_only: bool = True,
        concat_method: str = "channel",
        **wan_kwargs: Any,
    ):
        super().__init__()
        self.use_pi3 = use_pi3
        self.device = torch.device(device)
        self._pi3_shape: Optional[Tuple[int, int, int]] = None
        self._pi3_hw: Optional[Tuple[int, int]] = None
        if self.use_pi3:
            if pi3_checkpoint is None:
                self.pi3 = Pi3.from_pretrained(pi3_pretrained_id).to(self.device)
            else:
                self.pi3 = Pi3().to(self.device)
                if pi3_checkpoint.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    weight = load_file(pi3_checkpoint)
                else:
                    weight = torch.load(pi3_checkpoint, map_location=self.device, weights_only=False)
                # weights_only avoids executing pickled code; disable only if the checkpoint requires it.
                self.pi3.load_state_dict(weight)
            self.pi3.eval().requires_grad_(False)
        else:
            self.pi3 = None

        device_id = (self.device.index or 0) if self.device.type == "cuda" else 0
        self.wan = WanTI2V(
            config=wan_config,
            checkpoint_dir=wan_checkpoint_dir,
            device_id=device_id,
            rank=0,
            trainable=trainable_wan,
            convert_model_dtype=False,
            use_pi3_condition=self.use_pi3,
            concat_method=concat_method,
            **wan_kwargs,
        )
        if self.use_pi3:
            # Pi3 decoder latents concatenate the last two blocks, giving 2 * dec_embed_dim channels.
            # The adapter expects that full volume as input before projecting into the VAE latent space.
            pi3_channel_dim = 2 * self.pi3.dec_embed_dim
            self.wan.configure_pi3_adapters(
                pi3_channel_dim,
                patch_size=self.pi3.patch_size,
                patch_start_idx=self.pi3.patch_start_idx,
            )

    def _ensure_divisible_size(self, image):
        """
        Pad a PIL image so height and width are divisible by the Pi3 patch size.

        Args:
            image: PIL.Image to pad.

        Returns:
            PIL.Image with zero-padding applied on the right/bottom edges when needed.
        """
        patch = self.pi3.patch_size
        target_w = math.ceil(image.width / patch) * patch
        target_h = math.ceil(image.height / patch) * patch
        if target_w == image.width and target_h == image.height:
            return image
        padding = (0, 0, target_w - image.width, target_h - image.height)
        return TF.pad(image, padding, fill=0)

    def _pad_tensor_divisible(self, tensor: torch.Tensor) -> torch.Tensor:
        patch = self.pi3.patch_size
        leading_shape = tensor.shape[:-3]
        c, h, w = tensor.shape[-3:]
        target_h = math.ceil(h / patch) * patch
        target_w = math.ceil(w / patch) * patch
        if target_h == h and target_w == w:
            return tensor
        # F.pad pads in (left, right, top, bottom) order for 2D spatial dimensions.
        padded = F.pad(tensor.reshape(-1, c, h, w),
                       (0, target_w - w, 0, target_h - h))
        return padded.reshape(*leading_shape, c, target_h, target_w)

    def _prepare_image_inputs(self, image) -> Tuple[torch.Tensor, Any]:
        if isinstance(image, (str, Path)):
            tensor = load_images_as_tensor(str(image), interval=1)
            if tensor.numel() == 0:
                raise ValueError(
                    f"No image loaded from {image}; ensure the path points to a readable .png/.jpg/.jpeg file or directory."
                )
            image = tensor
        if not self.use_pi3:
            if isinstance(image, torch.Tensor):
                base = image
                if base.dim() == 4:
                    base = base[0]
                if base.dim() != 3:
                    raise ValueError(f"Expected image tensor with shape (C, H, W) or (N, C, H, W); got {tuple(image.shape)}")
                base_cpu = base if base.device.type == "cpu" else base.cpu()
                pil = TF.to_pil_image(base_cpu)
                tensor = base.unsqueeze(0).unsqueeze(0)
            else:
                pil = image if hasattr(image, "mode") and image.mode == "RGB" else image.convert("RGB")
                tensor = TF.to_tensor(pil).unsqueeze(0).unsqueeze(0)
            return tensor.to(self.device), pil
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            patch = self.pi3.patch_size
            needs_pad = (
                tensor.shape[-1] % patch != 0 or tensor.shape[-2] % patch != 0
            )
            if needs_pad:
                tensor = self._pad_tensor_divisible(tensor)
            pil = TF.to_pil_image(tensor[0, 0].cpu())
        else:
            pil = image if hasattr(image, "mode") and image.mode == "RGB" else image.convert("RGB")
            pil = self._ensure_divisible_size(pil)
            tensor = TF.to_tensor(pil).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), pil

    def _decode_pi3_latent_sequence(
        self,
        pi3_latent: torch.Tensor | None,
    ) -> Optional[Dict[str, Any]]:
        """
        Recover dynamic Pi3 latents (returned by diffusion) back to Pi3 head inputs and decode them per frame.

        Args:
            pi3_latent: Tensor of shape (C, F, H, W), (B, C, F, H, W), or frame-first flattened
                layout (F, B, H * W, C) representing the Pi3 latent slice produced by the diffusion
                model after the forward pass.

        Returns:
            Dict of decoded Pi3 predictions (points/conf/camera) or None when input is invalid.
        """
        print("Shape of input pi3 latents before decode", pi3_latent.shape)
        if not self.use_pi3 or self.pi3 is None:
            return None
        with torch.no_grad():
            recovered = pi3_latent
            if recovered is None:
                return None
            recovered = ensure_view_dim(recovered)
            print("Shape of recovered pi3 latents after ensure_view_dim", recovered.shape)
            # cached Pi3 shape stores register + spatial tokens; subtract registers to recover patch_h * patch_w.
            expected_hw = (
                None
                if self._pi3_shape is None
                else max(1, self._pi3_shape[1] - self.pi3.patch_start_idx)
            )
            # frame-first layout indicates recovered latents shaped as (F, B, H * W, 2 * dec_embed_dim)
            frame_first = (
                recovered.dim() == 4
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
                    # hw reflects patch grids (typically a few thousand tokens), so a short descending scan is cheap.
                    h = next(
                        (cand for cand in range(root, 0, -1) if hw % cand == 0),
                        1,
                    )
                    # Token counts are small; the descending scan returns the largest divisor <= sqrt(hw) to keep aspect ratio reasonable.
                    w = hw // h
                tokens = batch_first.reshape(b, f, hw, c)
            else:
                if recovered.dim() == 4:
                    recovered = recovered.unsqueeze(0)
                b, c, f, h, w = recovered.shape
                tokens = recovered.permute(0, 2, 3, 4, 1).reshape(b, f, h * w, c)
            
            print("Shape of tokens after operation", tokens.shape)
            register = torch.cat(
                [self.pi3.register_token, self.pi3.register_token],
                dim=-1,
            ).to(tokens.device, tokens.dtype)
            # Preserve batch/view and frame axes so downstream logging can report per-view shapes before flattening.
            register = register.repeat(b, f, 1, 1).reshape(
                b,
                f,
                self.pi3.patch_start_idx,
                tokens.shape[-1],
            )
            decoder_hidden = torch.cat([register, tokens], dim=2)
            tokens_len, embed_dim = decoder_hidden.shape[2], decoder_hidden.shape[3]
            decoder_hidden_flat = decoder_hidden.reshape(b * f, tokens_len, embed_dim)
            print(
                "Decoder hidden shapes: batch-first (B,F,tokens,C), frame-first (F,B,tokens,C), flattened (B*F,tokens,C)",
                decoder_hidden.shape,
                decoder_hidden.permute(1, 0, 2, 3).shape,
                decoder_hidden_flat.shape,
            )
            pos = self.pi3.position_getter(
                b * f, h, w, tokens.device)
            pos = pos + 1
            pos_special = torch.zeros(
                b * f,
                self.pi3.patch_start_idx,
                2,
                device=tokens.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

            H_pix = h * self.pi3.patch_size
            W_pix = w * self.pi3.patch_size
            decoded = self.pi3._decode_tokens(
                decoder_hidden,
                pos,
                H_pix,
                W_pix,
                b,
                f,
            )
            return decoded

    def generate_with_3d(self, prompt: str, image, enable_grad: bool = False, decode_pi3: bool = False, **kwargs) -> Dict[str, Any]:
        imgs, pil_image = self._prepare_image_inputs(image)
        video_condition = None
        # Reset per-call cache; used only to decode outputs from this generation.
        if self.use_pi3:
            with torch.no_grad():
                latents = self.pi3(imgs)
            latents = latents.copy()
            latents["patch_size"] = self.pi3.patch_size
            latents["patch_start_idx"] = self.pi3.patch_start_idx
            self._pi3_shape = latents["hidden"].shape
            self._pi3_hw = latents.get("hw")

            video_condition = latents
        if enable_grad:
            kwargs.setdefault("offload_model", False)
        generated = self.wan.generate(
            prompt,
            img=pil_image,
            video_condition=video_condition,
            enable_grad=enable_grad,
            return_latents=True,
            **kwargs,
        )
        if isinstance(generated, dict):
            video = generated.get("video")
            rgb_latent = generated.get("rgb_latent")
            pi3_latent = generated.get("pi3_latent")
        else:
            video = generated
            rgb_latent = None
            pi3_latent = None
        pi3_preds = None
        if decode_pi3 and pi3_latent is not None:
            print("Shape of pi3 latents before decode", pi3_latent.shape)
            pi3_preds = self._decode_pi3_latent_sequence(pi3_latent)
        return {
            "video": video,
            "rgb_latent": rgb_latent,
            "pi3_latent": pi3_latent,
            "pi3_preds": pi3_preds,
        }

    def training_step(
        self,
        prompt: str,
        image,
        gt_video: Optional[torch.Tensor] = None,
        gt_points: Optional[torch.Tensor] = None,
        gt_latent: Optional[torch.Tensor] = None,
        video_weight: float = 1.0,
        latent_weight: float = 1.0,
        point_weight: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        decode_pi3 = kwargs.pop("decode_pi3", gt_points is not None)
        outputs = self.generate_with_3d(
            prompt,
            image,
            enable_grad=True,
            decode_pi3=decode_pi3,
            **kwargs,
        )
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        if gt_latent is not None:
            if outputs.get('rgb_latent') is None:
                raise ValueError("Latent supervision requested but generation did not return rgb_latent.")
            target_latent = gt_latent.to(outputs['rgb_latent'].device, dtype=outputs['rgb_latent'].dtype)
            losses['latent'] = F.mse_loss(outputs['rgb_latent'], target_latent)
            total_loss = total_loss + latent_weight * losses['latent']
        if gt_video is not None:
            losses['video'] = F.mse_loss(outputs['video'], gt_video)
            total_loss = total_loss + video_weight * losses['video']
        if gt_points is not None:
            if not self.use_pi3:
                raise ValueError("Point supervision training requires Pi3 conditioning to be enabled. Set use_pi3=True to enable this feature.")
            losses['points'] = F.l1_loss(outputs['pi3_preds']['points'], gt_points)
            total_loss = total_loss + point_weight * losses['points']
        outputs['loss'] = total_loss
        outputs['losses'] = losses
        return outputs
