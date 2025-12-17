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
            weight = torch.load(
                pi3_checkpoint,
                map_location=self.device,
                weights_only=pi3_weights_only,
            )
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

    def generate_with_3d(self, prompt: str, image, enable_grad: bool = False, **kwargs) -> Dict[str, Any]:
        imgs, pil_image = self._prepare_image_inputs(image)
        with torch.no_grad():
            pi3_out = self.pi3(imgs, return_latents=True)
        latents = pi3_out['latents']
        latent_volume = self._build_latent_volume(latents)
        if enable_grad:
            kwargs.setdefault("offload_model", False)
        video = self.wan.generate(
            prompt,
            img=pil_image,
            video_condition=latent_volume,
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
