# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        use_pi3_condition=True,
        trainable=False,
        concat_method: str = "channel",
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.trainable = trainable
        self.latent_adapter = None
        self.use_pi3_condition = use_pi3_condition
        valid_concat = {"channel", "frame", "width"}
        if concat_method not in valid_concat:
            raise ValueError(
                f"Unsupported concat_method={concat_method}. Choose from {sorted(valid_concat)}."
            )
        self.concat_method = concat_method

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
            trainable=trainable)
        self._align_patch_embedding_for_conditioning()

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _align_patch_embedding_for_conditioning(self) -> None:
        """
        When PI3 conditioning is enabled, ensure the patch embedding expects concatenated
        RGB and conditioning channels by seeding the extra slice with the pretrained RGB weights.
        """
        if not self.use_pi3_condition or self.concat_method != "channel":
            return

        patch_embedding = getattr(self.model, "patch_embedding", None)
        if patch_embedding is None:
            return

        base_channels = self.vae.model.z_dim
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
        self.model.patch_embedding = new_patch

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, trainable=False):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)
        if trainable:
            model.train().requires_grad_(True)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 extra_context=None,
                 video_condition=None,
                 enable_grad=False,
                 return_latents: bool = False):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            extra_context (`Tensor`, *optional*, defaults to None):
                Additional conditioning tokens (B, L, C) appended to text embeddings, e.g. adapted Pi3 latents.
            video_condition (`Tensor`, *optional*, defaults to None):
                Additional conditioning video latents concatenated channel-wise with the encoded reference image.
            enable_grad (`bool`, *optional*, defaults to False):
                Enable gradient flow through the diffusion backbone for finetuning scenarios.
            return_latents (`bool`, *optional*, defaults to False):
                When True, also return the latent tensor produced by the diffusion loop.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                extra_context=extra_context,
                video_condition=video_condition,
                enable_grad=enable_grad,
                return_latents=return_latents)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
            extra_context=extra_context,
            video_condition=video_condition,
            enable_grad=enable_grad,
            return_latents=return_latents)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            extra_context=None,
            video_condition=None,
            enable_grad=False,
            return_latents: bool = False):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            extra_context (`Tensor`, *optional*, defaults to None):
                Additional conditioning tokens (B, L, C) appended to text embeddings, e.g. adapted Pi3 latents.
            video_condition (`Tensor`, *optional*, defaults to None):
                Additional conditioning video latents concatenated channel-wise with the encoded reference image.
            return_latents (`bool`, *optional*, defaults to False):
                When True, also return the latent tensor produced by the diffusion loop.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        frame_count = frame_num
        target_shape = (self.vae.model.z_dim,
                        (frame_count - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        grad_context = nullcontext() if enable_grad else torch.no_grad()

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                grad_context,
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

        if extra_context is not None:
            extra_context = extra_context.to(self.device)

        arg_c = {'context': context, 'seq_len': seq_len, 'extra_context': extra_context}
        arg_null = {'context': context_null, 'seq_len': seq_len, 'extra_context': extra_context}

        if offload_model or self.init_on_cpu:
            self.model.to(self.device)
            torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        if return_latents:
            return {"video": videos[0], "latent": x0[0]}
        return videos[0]

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            extra_context=None,
            video_condition=None,
            enable_grad=False,
            return_latents: bool = False):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            extra_context (`Tensor`, *optional*, defaults to None):
                Additional conditioning tokens (B, L, C) appended to text embeddings, e.g. adapted Pi3 latents.
            video_condition (`Tensor`, *optional*, defaults to None):
                Extra conditioning video latents concatenated with the encoded reference image along channels.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        frame_count = frame_num
        seq_len = ((frame_count - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        if not self.use_pi3_condition:
            video_condition = None

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])
        cond_latent = z[0]
        fused_latent = cond_latent
        pi3_condition_adapted = None
        channel_count = cond_latent.shape[0]
        condition_channels = 0
        latent_frames = (frame_count - 1) // self.vae_stride[0] + 1
        concat_method = getattr(self, "concat_method", "channel")
        if video_condition is not None:
            cond = video_condition
            if isinstance(cond, list):
                cond = cond[0]
            if cond.dim() not in (4, 5):
                raise ValueError(
                    "video_condition must have shape (C, F, H, W) or (B, C, F, H, W)."
                )
            if cond.dim() == 5:
                cond = cond[0]
            cond = cond.to(device=self.device, dtype=cond_latent.dtype)
            if cond.shape[1:] != cond_latent.shape[1:]:
                # Preserve temporal length for frame-wise fusion; only match spatial dims.
                target_size = cond_latent.shape[1:] if concat_method != "frame" else (
                    cond.shape[1],
                    cond_latent.shape[2],
                    cond_latent.shape[3],
                )
                cond = torch.nn.functional.interpolate(
                    cond.unsqueeze(0),
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            cond_c, cond_f, cond_h, cond_w = cond.shape
            latent_c, latent_f, latent_h, latent_w = cond_latent.shape
            can_project = (
                self.latent_adapter is not None
                and cond_c == self.latent_adapter.in_channels
                and (cond_f, cond_h, cond_w) == (latent_f, latent_h, latent_w)
            )
            if can_project:
                cond = self.latent_adapter(cond.unsqueeze(0)).squeeze(0)
            pi3_condition_adapted = cond
            if concat_method == "channel":
                fused_latent = torch.cat([cond_latent, cond], dim=0)
            elif concat_method == "frame":
                fused_latent = torch.cat([cond_latent, cond], dim=1)
            elif concat_method == "width":
                fused_latent = torch.cat([cond_latent, cond], dim=3)
            else:
                raise ValueError(f"Unsupported concat_method: {concat_method}")
            condition_channels = cond.shape[0]

        latent_h = fused_latent.shape[2]
        latent_w = fused_latent.shape[3]

        # Recompute sequence length from the fused latent dimensions (frames × spatial patches);
        # fused_latent falls back to cond_latent when no Pi3 conditioning is provided.
        seq_len = int(math.ceil(
            (latent_h * latent_w) /
            (self.patch_size[1] * self.patch_size[2]) *
            latent_frames / self.sp_size)) * self.sp_size


        noise = torch.randn(
            fused_latent.shape[0],
            latent_frames,
            latent_h,
            latent_w,
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        grad_context = nullcontext() if enable_grad else torch.no_grad()

        # define variables avoid undefined reference error
        final_latent = None
        rgb_latent = None
        pi3_latent = None
        x0 = None

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                grad_context,
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * fused_latent + mask2[0] * latent

            if extra_context is not None:
                extra_context = extra_context.to(self.device)
            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
                'extra_context': extra_context,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
                'extra_context': extra_context,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                if pi3_condition_adapted is not None and noise_pred.shape[0] != fused_latent.shape[0]:
                    channel_diff = fused_latent.shape[0] - noise_pred.shape[0]
                    if channel_diff > 0:
                        pad_shape = (channel_diff, *noise_pred.shape[1:])
                        if channel_diff == condition_channels and noise_pred.shape[0] >= condition_channels:
                            channel_fill = noise_pred[:condition_channels].clone()
                        else:
                            channel_fill = torch.zeros(
                                pad_shape, device=noise_pred.device, dtype=noise_pred.dtype)
                        noise_pred = torch.cat([noise_pred, channel_fill], dim=0)
                    else:
                        noise_pred = noise_pred[:fused_latent.shape[0]]

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * fused_latent + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if video_condition is not None:
                final_latent = latent
                if concat_method == "channel":
                    rgb_latent = final_latent[:channel_count]
                    pi3_latent = (
                        final_latent[channel_count:channel_count + condition_channels]
                        if condition_channels > 0 else None
                    )
                    output_latent = rgb_latent
                elif concat_method == "frame":
                    # Decode with the fused latent so the frame dimension reflects the concatenation.
                    rgb_latent = final_latent[:, :cond_latent.shape[1], :, :]
                    pi3_latent = final_latent[:, cond_latent.shape[1]:, :, :]
                    output_latent = final_latent
                elif concat_method == "width":
                    # Decode with the fused latent so the spatial width reflects the concatenation.
                    rgb_latent = final_latent[:, :, :, :cond_latent.shape[3]]
                    pi3_latent = final_latent[:, :, :, cond_latent.shape[3]:]
                    output_latent = final_latent
                else:
                    pi3_latent = None
                    rgb_latent = final_latent
                    output_latent = final_latent
                x0 = [output_latent]
            else:
                x0 = [latent]

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        output_video = videos[0] if self.rank == 0 else None
        output_pi3_latent = pi3_latent if self.rank == 0 else None
        output_rgb_latent = rgb_latent if self.rank == 0 else None

        del noise, final_latent, rgb_latent, pi3_latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        if return_latents:
            result = {
                "video": output_video,
                "rgb_latent": output_rgb_latent,
            }
            if output_pi3_latent is not None:
                result["pi3_latent"] = output_pi3_latent
            return result
        return output_video
