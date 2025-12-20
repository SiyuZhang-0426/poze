import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pi3.models.pi3 import Pi3
from .textimage2video import WanTI2V


class Pi3GuidedTI2V(nn.Module):
    """
    Pipeline that freezes Pi3, adapts its latents, and optionally finetunes Wan2.2 TI2V.
    """

    def __init__(
        self,
        wan_config,
        wan_checkpoint_dir: str,
        pi3_checkpoint: Optional[str] = None,
        device: str = "cuda",
        trainable_wan: bool = False,
        pi3_pretrained_id: str = "yyfz233/Pi3",
        pi3_weights_only: bool = True,
        **wan_kwargs: Any,
    ):
        super().__init__()
        self.device = torch.device(device)
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

        device_id = (self.device.index or 0) if self.device.type == "cuda" else 0
        self.wan = WanTI2V(
            config=wan_config,
            checkpoint_dir=wan_checkpoint_dir,
            device_id=device_id,
            rank=0,
            trainable=trainable_wan,
            convert_model_dtype=False,
            **wan_kwargs,
        )
        # Pi3 decoder latents concatenate the last two blocks, giving 2 * dec_embed_dim channels.
        # The adapter expects that full volume as input before projecting into the VAE latent space.
        pi3_channel_dim = 2 * self.pi3.dec_embed_dim
        self.latent_adapter = nn.Conv3d(
            in_channels=pi3_channel_dim,
            out_channels=self.wan.vae.model.z_dim,
            kernel_size=1,
        ).to(self.device)
        # Recover Pi3 latents (from diffusion output) back into decoder token space for per-frame decoding.
        self.pi3_recover_adapter = nn.Conv3d(
            in_channels=self.wan.vae.model.z_dim,
            out_channels=pi3_channel_dim,
            kernel_size=1,
        ).to(self.device)
        with torch.no_grad():
            # Seed with a channel-copy identity so image latents stay intact while Pi3 features are blended in.
            self.latent_adapter.weight.zero_()
            if self.latent_adapter.bias is not None:
                self.latent_adapter.bias.zero_()
            shared = min(
                self.latent_adapter.in_channels,
                self.latent_adapter.out_channels,
            )
            for i in range(shared):
                self.latent_adapter.weight[i, i, 0, 0, 0] = 1.0
            # Mirror the identity-style init for the recovery adapter so round-tripping is stable before finetuning.
            self.pi3_recover_adapter.weight.zero_()
            if self.pi3_recover_adapter.bias is not None:
                self.pi3_recover_adapter.bias.zero_()
            shared_recover = min(
                self.pi3_recover_adapter.in_channels,
                self.pi3_recover_adapter.out_channels,
            )
            for i in range(shared_recover):
                self.pi3_recover_adapter.weight[i, i, 0, 0, 0] = 1.0
        self.wan.latent_adapter = self.latent_adapter
        self._align_patch_embedding_for_pi3()

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

    def _build_latent_volume(self, latents: Dict[str, Any]) -> torch.Tensor:
        """
        Reshape Pi3 decoder hidden tokens into a spatial volume aligned with the patch grid.

        Args:
            latents: Pi3 latent dictionary containing 'decoder_hidden', 'hw', 'batch', and 'frames'.

        Returns:
            torch.Tensor: A tensor shaped (B, C, F, H, W) representing per-frame spatial latents.
        """
        patch_h = latents['hw'][0] // self.pi3.patch_size
        patch_w = latents['hw'][1] // self.pi3.patch_size
        tokens = latents['decoder_hidden'][:, self.pi3.patch_start_idx:, :]
        tokens = tokens.view(
            latents['batch'],
            latents['frames'],
            patch_h,
            patch_w,
            tokens.size(-1),
        )
        return tokens.permute(0, 4, 1, 2, 3).contiguous()

    def _decode_pi3_latent_sequence(self, pi3_latent: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Recover dynamic Pi3 latents (returned by diffusion) back to Pi3 head inputs and decode them per frame.

        Args:
            pi3_latent: Tensor of shape (C, F, H, W) or (B, C, F, H, W) representing the Pi3 latent slice
                produced by the diffusion model after the forward pass.

        Returns:
            Dict of decoded Pi3 predictions (points/conf/camera) or None when input is invalid.
        """
        if pi3_latent is None:
            return None
        if isinstance(pi3_latent, list):
            if len(pi3_latent) == 0:
                return None
            pi3_latent = pi3_latent[0]
        if not isinstance(pi3_latent, torch.Tensor):
            return None

        if pi3_latent.dim() == 4:
            pi3_latent = pi3_latent.unsqueeze(0)
        if pi3_latent.dim() != 5:
            return None

        with torch.no_grad():
            recovered = self.pi3_recover_adapter(pi3_latent.to(self.device))
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

    def generate_with_3d(self, prompt: str, image, enable_grad: bool = False, **kwargs) -> Dict[str, Any]:
        imgs, pil_image = self._prepare_image_inputs(image)
        with torch.no_grad():
            pi3_out = self.pi3(imgs, return_latents=True)
        latents = pi3_out['latents']
        latent_volume = self._build_latent_volume(latents)
        if enable_grad:
            kwargs.setdefault("offload_model", False)
        generated = self.wan.generate(
            prompt,
            img=pil_image,
            video_condition=latent_volume,
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
        # pi3_preds = self._decode_pi3_latent_sequence(pi3_latent)
        return {
            "video": video,
            "rgb_latent": rgb_latent,
            # "pi3_preds": pi3_preds,
        }

    def training_step(
        self,
        prompt: str,
        image,
        gt_video: Optional[torch.Tensor] = None,
        gt_points: Optional[torch.Tensor] = None,
        video_weight: float = 1.0,
        point_weight: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.generate_with_3d(prompt, image, enable_grad=True, **kwargs)
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        if gt_video is not None:
            losses['video'] = F.mse_loss(outputs['video'], gt_video)
            total_loss = total_loss + video_weight * losses['video']
        if gt_points is not None:
            losses['points'] = F.l1_loss(outputs['pi3']['points'], gt_points)
            total_loss = total_loss + point_weight * losses['points']
        outputs['loss'] = total_loss
        outputs['losses'] = losses
        return outputs
