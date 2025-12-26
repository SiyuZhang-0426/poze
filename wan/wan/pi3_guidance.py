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
        self._last_pi3_target_size: Optional[Tuple[int, int, int]] = None
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

    def _align_patch_embedding_for_pi3(self) -> None:
        """
        If the Wan patch embedding expects concatenated RGB + Pi3 channels, seed the
        Pi3 portion with the pretrained RGB weights to avoid random initialization.
        """
        patch_embedding = getattr(self.wan.model, "patch_embedding", None)
        if patch_embedding is None:
            # Some alternative model wrappers may replace or omit the patch embedding entirely.
            return

        base_channels = self.wan.vae.model.z_dim
        # Patch embedding expects RGB latents plus Pi3 latents; the adapter produces the same channel count as RGB.
        expected_channels = base_channels * 2
        weight = getattr(patch_embedding, "weight", None)
        if weight is None or weight.dim() != 5:
            return

        if patch_embedding.in_channels not in (base_channels, expected_channels):
            return

        if patch_embedding.in_channels == expected_channels:
            with torch.no_grad():
                rgb_weights = weight[:, :base_channels].clone()
                weight[:, base_channels:expected_channels] = rgb_weights
            return

        if patch_embedding.in_channels == base_channels:
            bias = patch_embedding.bias
            new_patch = nn.Conv3d(
                in_channels=expected_channels,
                out_channels=patch_embedding.out_channels,
                kernel_size=patch_embedding.kernel_size,
                stride=patch_embedding.stride,
                padding=patch_embedding.padding,
                dilation=patch_embedding.dilation,
                groups=patch_embedding.groups,
                bias=bias is not None,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                new_patch.weight.zero_()
                rgb_weights = weight.clone()
                new_patch.weight[:, :base_channels] = rgb_weights
                new_patch.weight[:, base_channels:expected_channels] = rgb_weights
                if bias is not None:
                    new_patch.bias.copy_(bias)
            self.wan.model.patch_embedding = new_patch

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
        pi3_latent: torch.Tensor,
        target_size: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Recover dynamic Pi3 latents (returned by diffusion) back to Pi3 head inputs and decode them per frame.

        Args:
            pi3_latent: Tensor of shape (C, F, H, W) or (B, C, F, H, W) representing the Pi3 latent slice
                produced by the diffusion model after the forward pass.

        Returns:
            Dict of decoded Pi3 predictions (points/conf/camera) or None when input is invalid.
        """
        if not self.use_pi3 or self.pi3 is None:
            return None
        resolved_target_size = target_size or self._last_pi3_target_size
        with torch.no_grad():
            recovered = self.wan.preprocess_pi3_latent(pi3_latent, resolved_target_size)
            if recovered is None:
                return None
            if recovered.dim() == 4:
                recovered = recovered.unsqueeze(0)
            b, c, f, h, w = recovered.shape
            # Rebuild decoder_hidden tokens (without registers) then prepend doubled register tokens.
            tokens = recovered.permute(0, 2, 3, 4, 1).reshape(b * f, h * w, c)
            register = torch.cat(
                [self.pi3.register_token, self.pi3.register_token],
                dim=-1,
            ).to(tokens.device, tokens.dtype)
            register = register.repeat(b, f, 1, 1).reshape(
                b * f,
                self.pi3.patch_start_idx,
                c,
            )
            decoder_hidden = torch.cat([register, tokens], dim=1)

            pos = self.pi3.position_getter(
                b * f, h, w, tokens.device).to(tokens.dtype)
            pos = pos + 1
            pos_special = torch.zeros(
                b * f,
                self.pi3.patch_start_idx,
                2,
                device=tokens.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

            point_tokens = self.pi3.point_decoder(decoder_hidden, xpos=pos)
            conf_tokens = self.pi3.conf_decoder(decoder_hidden, xpos=pos)
            camera_tokens = self.pi3.camera_decoder(decoder_hidden, xpos=pos)

            H_pix = h * self.pi3.patch_size
            W_pix = w * self.pi3.patch_size
            decoded = self.pi3._decode_tokens(
                point_tokens,
                conf_tokens,
                camera_tokens,
                H_pix,
                W_pix,
                b,
                f,
            )
            decoded['latents'] = dict(
                decoder_hidden=decoder_hidden,
                point_tokens=point_tokens,
                conf_tokens=conf_tokens,
                camera_tokens=camera_tokens,
                hw=(H_pix, W_pix),
                frames=f,
                batch=b,
            )
            return decoded

    def generate_with_3d(self, prompt: str, image, enable_grad: bool = False, decode_pi3: bool = False, **kwargs) -> Dict[str, Any]:
        imgs, pil_image = self._prepare_image_inputs(image)
        video_condition = None
        pi3_target_size = None
        # Reset per-call cache; used only to decode outputs from this generation.
        self._last_pi3_target_size = None
        if self.use_pi3:
            with torch.no_grad():
                latents = self.pi3(imgs)
            latents = latents.copy()
            latents["patch_size"] = self.pi3.patch_size
            latents["patch_start_idx"] = self.pi3.patch_start_idx
            patch_h = latents['hw'][0] // self.pi3.patch_size
            patch_w = latents['hw'][1] // self.pi3.patch_size
            pi3_target_size = (latents['frames'], patch_h, patch_w)
            self._last_pi3_target_size = pi3_target_size
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
            processed_pi3_latent = pi3_latent
        else:
            video = generated
            rgb_latent = None
            processed_pi3_latent = None
        pi3_preds = None
        # if decode_pi3 and processed_pi3_latent is not None:
        #     pi3_preds = self._decode_pi3_latent_sequence(
        #         processed_pi3_latent,
        #         target_size=pi3_target_size,
        #     )
        return {
            "video": video,
            "rgb_latent": rgb_latent,
            "pi3_latent": processed_pi3_latent,
            "pi3": pi3_preds,
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
            losses['points'] = F.l1_loss(outputs['pi3']['points'], gt_points)
            total_loss = total_loss + point_weight * losses['points']
        outputs['loss'] = total_loss
        outputs['losses'] = losses
        return outputs
