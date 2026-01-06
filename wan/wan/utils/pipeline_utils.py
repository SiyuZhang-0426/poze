import logging
import types
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..distributed.sequence_parallel import sp_attn_forward, sp_dit_forward


def configure_model(
    model,
    *,
    use_sp: bool,
    dit_fsdp: bool,
    shard_fn,
    convert_model_dtype: bool,
    init_on_cpu: bool,
    device: torch.device,
    param_dtype: torch.dtype,
    trainable: bool = False,
):
    """
    Configure a model object for inference or training with optional SP/FSDP.
    """
    model.eval().requires_grad_(False)
    if trainable:
        model.train().requires_grad_(True)

    if use_sp:
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(sp_attn_forward,
                                                       block.self_attn)
        model.forward = types.MethodType(sp_dit_forward, model)

    if dist.is_initialized():
        dist.barrier()

    if dit_fsdp:
        model = shard_fn(model)
    else:
        if convert_model_dtype:
            model.to(param_dtype)
        if not init_on_cpu:
            model.to(device)

    return model


def prepare_model_for_timestep(
    t: torch.Tensor,
    boundary: int,
    *,
    low_noise_model,
    high_noise_model,
    offload_model: bool,
    init_on_cpu: bool,
    device: torch.device,
):
    """
    Select and return the correct model for the current timestep, handling optional offload.
    """
    if t.item() >= boundary:
        required_model = high_noise_model
        offload_target = low_noise_model
    else:
        required_model = low_noise_model
        offload_target = high_noise_model

    if offload_model or init_on_cpu:
        if next(offload_target.parameters()).device.type == 'cuda':
            offload_target.to('cpu')
        if next(required_model.parameters()).device.type == 'cpu':
            required_model.to(device)

    return required_model


def align_patch_embedding_for_conditioning(
    model,
    base_channels: int,
    use_pi3_condition: bool,
    concat_method: str,
):
    """
    Ensure the patch embedding expects concatenated RGB and conditioning channels.
    """
    if not use_pi3_condition or concat_method != "channel":
        return

    patch_embedding = getattr(model, "patch_embedding", None)
    if patch_embedding is None:
        return

    weight = getattr(patch_embedding, "weight", None)
    if weight is None or weight.dim() != 5:
        return

    expected_channels = base_channels * 2
    if patch_embedding.in_channels not in (base_channels, expected_channels):
        return

    if patch_embedding.in_channels == expected_channels:
        with torch.no_grad():
            rgb_weights = weight[:, :base_channels].clone()
            weight[:, base_channels:expected_channels] = rgb_weights
        return

    bias = patch_embedding.bias
    new_patch = torch.nn.Conv3d(
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
    model.patch_embedding = new_patch


def create_pi3_adapters(pi3_channel_dim: int, target_channels: int,
                        device: torch.device):
    latent_adapter = torch.nn.Conv3d(
        in_channels=pi3_channel_dim,
        out_channels=target_channels,
        kernel_size=1,
        device=device,
    )
    pi3_recover_adapter = torch.nn.Conv3d(
        in_channels=target_channels,
        out_channels=pi3_channel_dim,
        kernel_size=1,
        device=device,
    )
    with torch.no_grad():
        latent_adapter.weight.zero_()
        if latent_adapter.bias is not None:
            latent_adapter.bias.zero_()
        shared = min(latent_adapter.in_channels, latent_adapter.out_channels)
        for i in range(shared):
            latent_adapter.weight[i, i, 0, 0, 0] = 1.0

        pi3_recover_adapter.weight.zero_()
        if pi3_recover_adapter.bias is not None:
            pi3_recover_adapter.bias.zero_()
        shared_recover = min(
            pi3_recover_adapter.in_channels,
            pi3_recover_adapter.out_channels,
        )
        for i in range(shared_recover):
            pi3_recover_adapter.weight[i, i, 0, 0, 0] = 1.0

    return latent_adapter, pi3_recover_adapter


def build_pi3_latent_volume(
    latents: dict,
    *,
    default_patch_size: int | None,
    default_patch_start_idx: int | None,
) -> torch.Tensor:
    patch_size = latents.get("patch_size", default_patch_size)
    patch_start_idx = latents.get("patch_start_idx", default_patch_start_idx)
    if patch_size is None:
        raise ValueError(
            "patch_size must be provided to build Pi3 latent volume (set via configure_pi3_adapters or latents['patch_size'])."
        )
    if patch_start_idx is None:
        raise ValueError(
            "patch_start_idx must be provided to build Pi3 latent volume (set via configure_pi3_adapters or latents['patch_start_idx'])."
        )

    patch_h = latents['hw'][0] // patch_size
    patch_w = latents['hw'][1] // patch_size
    tokens = latents['hidden'][:, patch_start_idx:, :]
    tokens = tokens.view(
        latents['batch'],
        latents['frames'],
        patch_h,
        patch_w,
        tokens.size(-1),
    )
    return tokens.permute(0, 4, 1, 2, 3).contiguous()


def prepare_pi3_condition(
    video_condition,
    cond_latent: torch.Tensor,
    *,
    device: torch.device,
    latent_adapter,
    use_pi3_condition: bool,
    concat_method: str,
    default_patch_size: int | None,
    default_patch_start_idx: int | None,
) -> Tuple[torch.Tensor | None, int]:
    if video_condition is None or not use_pi3_condition:
        return None, 0

    cond = video_condition
    if isinstance(cond, dict):
        cond = build_pi3_latent_volume(
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
        raise ValueError(
            "video_condition must have shape (C, F, H, W) or (B, C, F, H, W).")
    cond = cond.to(device=device, dtype=cond_latent.dtype)

    target_size = (cond_latent.shape[1], cond_latent.shape[2],
                   cond_latent.shape[3])
    if cond.shape[-3:] != target_size:
        cond = torch.nn.functional.interpolate(
            cond, size=target_size, mode="trilinear", align_corners=False)

    conv_output = latent_adapter(cond) if latent_adapter is not None else cond
    conv_output = conv_output.contiguous()

    logging.debug("Prepared Pi3 condition with shape %s", conv_output.shape)
    return conv_output.squeeze(0), conv_output.shape[1]


def recover_pi3_latents(
    pi3_latent: torch.Tensor | list[torch.Tensor] | None,
    target_size: tuple[int, int, int],
    *,
    recover_adapter=None,
    flatten_to_frames: bool = False,
):
    if pi3_latent is None:
        return None
    if isinstance(pi3_latent, list):
        if len(pi3_latent) == 0:
            return None
        pi3_latent = pi3_latent[0]

    logging.debug("Pi3 latent before recovery shape: %s",
                  getattr(pi3_latent, "shape", None))

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
            processed_latent = processed_latent.repeat(1, repeat_factor, 1, 1,
                                                       1)
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


def pi3_recovery_dims(
    video_condition: dict,
    pi3_latent: torch.Tensor,
    videos,
    *,
    default_patch_size: int | None,
) -> tuple[int, int, int]:
    patch_size = video_condition.get("patch_size", default_patch_size)
    patch_size = max(patch_size or 1, 1)
    hw = video_condition.get("hw")
    if hw is None or len(hw) < 2:
        patch_h = max(1, pi3_latent.shape[-2])
        patch_w = max(1, pi3_latent.shape[-1])
    else:
        patch_h = max(1, hw[0] // patch_size)
        patch_w = max(1, hw[1] // patch_size)
    has_video = isinstance(videos, (list, tuple)) and len(videos) > 0
    target_frames = pi3_latent.shape[1]
    if has_video:
        try:
            video_shape = tuple(getattr(videos[0], "shape", ()))
            if len(video_shape) == 5 and video_shape[2] > 0:
                target_frames = video_shape[2]
            elif len(video_shape) >= 2 and video_shape[1] > 0:
                target_frames = video_shape[1]
        except Exception:
            target_frames = pi3_latent.shape[1]
    return target_frames, patch_h, patch_w
