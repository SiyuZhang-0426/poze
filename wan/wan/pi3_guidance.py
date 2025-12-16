import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from pi3.models.pi3 import Pi3
from .textimage2video import WanTI2V


class Pi3FeatureAdapter(nn.Module):
    """
    Simple adaptive layer to reduce Pi3 decoder tokens and align them with Wan's text embedding space.
    """

    def __init__(self, in_dim: int, out_dim: int, target_tokens: int = 64):
        super().__init__()
        side = max(1, int(math.sqrt(target_tokens)))
        self.target_hw = (side, max(1, target_tokens // side))
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            tokens: (B*N, L, C) Pi3 decoder hidden tokens (register tokens should be stripped).
            patch_hw: (patch_h, patch_w) grid size before tokenization.
        """
        ph, pw = patch_hw
        tokens = tokens[:, :ph * pw, :]
        grid = tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(-1), ph, pw)
        pooled = F.adaptive_avg_pool2d(grid, self.target_hw)
        pooled = pooled.flatten(2).transpose(1, 2)
        return self.proj(pooled)


class Pi3GuidedTI2V(nn.Module):
    """
    Pipeline that freezes Pi3, adapts its latents, and optionally finetunes Wan2.2 TI2V.
    """

    def __init__(
        self,
        wan_config,
        wan_checkpoint_dir: str,
        pi3_checkpoint: Optional[str] = None,
        adapter_tokens: int = 64,
        device: str = "cuda",
        trainable_wan: bool = False,
        **wan_kwargs: Any,
    ):
        super().__init__()
        self.device = torch.device(device)
        if pi3_checkpoint is None:
            self.pi3 = Pi3.from_pretrained("yyfz233/Pi3").to(self.device)
        else:
            self.pi3 = Pi3().to(self.device)
            weight = torch.load(pi3_checkpoint, map_location=self.device, weights_only=True)
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
        self.adapter = Pi3FeatureAdapter(
            in_dim=self.pi3.dec_embed_dim * 2,
            out_dim=self.wan.model.dim,
            target_tokens=adapter_tokens,
        ).to(self.device)

    def _prepare_image_inputs(self, image) -> Tuple[torch.Tensor, Any]:
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            pil = TF.to_pil_image(tensor[0, 0].cpu())
        else:
            pil = image
            tensor = TF.to_tensor(image).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), pil

    def _build_guidance(self, latents: Dict[str, Any]) -> torch.Tensor:
        patch_h = latents['hw'][0] // self.pi3.patch_size
        patch_w = latents['hw'][1] // self.pi3.patch_size
        tokens = latents['decoder_hidden'][:, self.pi3.patch_start_idx:, :]
        tokens = tokens.view(latents['batch'], latents['frames'], tokens.size(1), tokens.size(2)).mean(dim=1)
        return self.adapter(tokens, (patch_h, patch_w))

    def generate_with_3d(self, prompt: str, image, enable_grad: bool = False, **kwargs) -> Dict[str, Any]:
        imgs, pil_image = self._prepare_image_inputs(image)
        with torch.no_grad():
            pi3_out = self.pi3(imgs, return_latents=True)
        latents = pi3_out['latents']
        guidance = self._build_guidance(latents)
        if enable_grad:
            kwargs.setdefault("offload_model", False)
        video = self.wan.generate(
            prompt,
            img=pil_image,
            extra_context=guidance,
            enable_grad=enable_grad,
            **kwargs,
        )
        pi3_preds = self.pi3.decode_from_latents(latents)
        return {"video": video, "pi3": pi3_preds, "latents": latents}

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
