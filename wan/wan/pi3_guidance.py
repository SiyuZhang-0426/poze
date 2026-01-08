from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi3.models.pi3 import Pi3
from .pi3_layers import Pi3RecoverLayer, Pi3StitchingLayer, prepare_image_inputs
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
        use_pi3: bool = True,
        device: str = "cuda",
        trainable_wan: bool = False,
        pi3_pretrained_id: str = "yyfz233/Pi3",
        concat_method: str = "channel",
        **wan_kwargs: Any,
    ):
        super().__init__()
        self.use_pi3 = use_pi3
        self.device = torch.device(device)
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
                self.pi3.load_state_dict(weight)
            self.pi3.eval().requires_grad_(False)
        else:
            self.pi3 = None
        
        self.stitching_layer = Pi3StitchingLayer(device=self.device)
        self.recover_layer = Pi3RecoverLayer(self.pi3, device=self.device)

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
            pi3_recover_layer=self.recover_layer,
            pi3_stitching_layer=self.stitching_layer,
            **wan_kwargs,
        )
        if self.use_pi3:
            pi3_embed_dim = 2 * self.pi3.dec_embed_dim
            target_channels = self.wan.vae.model.z_dim
            self.stitching_layer.configure_condition_adapter(
                pi3_channel_dim=pi3_embed_dim,
                target_channels=target_channels,
                device=self.device,
                default_patch_size=self.pi3.patch_size,
                default_patch_start_idx=self.pi3.patch_start_idx,
            )
            self.recover_layer.configure_recover_adapter(
                pi3_channel_dim=pi3_embed_dim,
                target_channels=target_channels,
                device=self.device,
            )

    def forward(
        self,
        prompt: str,
        image,
        enable_grad: bool = False,
        decode_pi3: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        return self.generate_with_3d(
            prompt,
            image,
            enable_grad=enable_grad,
            decode_pi3=decode_pi3,
            **kwargs,
        )

    def generate_with_3d(
        self,
        prompt: str,
        image,
        enable_grad: bool = False,
        decode_pi3: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        imgs, pil_image = prepare_image_inputs(self.pi3, image, self.use_pi3, self.device)
        video_condition = None
        if self.use_pi3:
            with torch.no_grad():
                latents = self.pi3(imgs)
            self.recover_layer.cache_latent_metadata(latents)
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
            pi3_latent = generated.get("pi3_latent")
        else:
            video = generated
            pi3_latent = None
        pi3_preds = None
        logging.info(f"pi3_latent shape before recover: {pi3_latent.shape if pi3_latent is not None else None}")
        if decode_pi3 and pi3_latent is not None:
            pi3_preds = self.recover_layer(
                mode="decode",
                pi3_latent=pi3_latent,
                latents=None,
                cache_metadata=False,
            )
        return {
            "video": video,
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
        outputs = self.forward(
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
