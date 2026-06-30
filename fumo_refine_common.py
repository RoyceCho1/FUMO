#!/usr/bin/env python
"""Shared runtime utilities for FUMO refine validation and inference."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from basicsr.models.archs.NAFNet_arch import NAFNet
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel

from diffusion.controlnetvae import ControlNetVAEModel
from diffusion.pipeline_onestep import OneStepPipeline
from fumo_mlocal import load_map_pil, resolve_m_local_path
from wavelet_color_fix import wavelet_decomposition


FINAL_SCORE_SSIM_WEIGHT = 10.0
FINAL_SCORE_LPIPS_WEIGHT = 5.0


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def first_existing(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_jsonl(path: str | Path) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    return torch.float32


def _normalize_to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_hf_image(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    return _normalize_to_01(high_freq)


def compute_hf_mag(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (hf_mag / mean).clamp(0.0, 1.0)


def load_prior_tensor(prior_path: str | Path) -> torch.Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    return TF.to_tensor(prior_img)


def final_score(psnr_value: float, ssim_value: float, lpips_value: float) -> float:
    return psnr_value + FINAL_SCORE_SSIM_WEIGHT * ssim_value - FINAL_SCORE_LPIPS_WEIGHT * lpips_value


def rgb_to_y_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}.")
    r = image[:, 0:1]
    g = image[:, 1:2]
    b = image[:, 2:3]
    return 0.256789 * r + 0.504129 * g + 0.097906 * b + 16.0 / 255.0


def calculate_validation_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_metric) -> dict:
    pred = pred.clamp(0.0, 1.0)
    target = target.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    pred_y = rgb_to_y_tensor(pred)
    target_y = rgb_to_y_tensor(target)
    mse = (pred_y - target_y).pow(2).flatten(1).mean(dim=1).clamp(min=1e-12)
    psnr_value = (20.0 * torch.log10(1.0 / torch.sqrt(mse))).mean().item()

    from utils.loss_utils import ssim

    ssim_value = ssim(pred_y, target_y).item()
    with torch.no_grad():
        lpips_value = lpips_metric(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()

    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "final_score": final_score(psnr_value, ssim_value, lpips_value),
    }


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    return TF.to_pil_image(image)


def labeled_image_grid(images: List[Image.Image], labels: List[str]) -> Image.Image:
    assert len(images) == len(labels)
    w, h = images[0].size
    label_height = max(36, h // 18)
    grid = Image.new("RGB", size=(len(images) * w, h + label_height), color="white")
    for idx, image in enumerate(images):
        grid.paste(image.convert("RGB"), box=(idx * w, 0))

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, w // 36))
    except OSError:
        font = ImageFont.load_default()

    for idx, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = idx * w + (w - text_w) // 2
        y = h + (label_height - text_h) // 2
        draw.text((x, y), label, fill="black", font=font)
    return grid


class ValidationJsonlDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        resolution: int,
        resolution_mode: str = "square",
        resize: Tuple[int, int] | None = None,
        num_images: int | None = None,
        load_m_local: bool = False,
        m_local_dirs: List[str | Path] | None = None,
        m_local_column: str = "m_local",
        m_local_missing_policy: str = "error",
    ):
        if num_images is not None:
            entries = entries[: int(num_images)]
        self.entries = entries
        self.resolution = resolution
        self.resolution_mode = resolution_mode
        self.resize = tuple(resize) if resize is not None else None
        self.load_m_local = load_m_local
        self.m_local_dirs = m_local_dirs
        self.m_local_column = m_local_column
        self.m_local_missing_policy = m_local_missing_policy

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        item = self.entries[idx]
        cond = Image.open(item["conditioning_image"]).convert("RGB")
        gt = Image.open(item["image"]).convert("RGB")
        prior = TF.to_pil_image(load_prior_tensor(item["prior"]))
        if self.load_m_local:
            resolved_m_local_path = resolve_m_local_path(
                item,
                item["conditioning_image"],
                self.m_local_dirs,
                self.m_local_column,
                self.m_local_missing_policy,
            )
            m_local = Image.new("L", cond.size, color=0) if resolved_m_local_path is None else load_map_pil(resolved_m_local_path)
        else:
            m_local = None

        if self.resolution_mode == "square":
            target_size = (self.resolution, self.resolution)
        elif self.resize is not None:
            target_size = self.resize
        else:
            target_size = cond.size

        if cond.size != target_size:
            cond = cond.resize(target_size, Image.Resampling.BILINEAR)
        if gt.size != target_size:
            gt = gt.resize(target_size, Image.Resampling.BILINEAR)
        if prior.size != target_size:
            prior = prior.resize(target_size, Image.Resampling.LANCZOS)
        if m_local is not None and m_local.size != target_size:
            m_local = m_local.resize(target_size, Image.Resampling.LANCZOS)

        result = {
            "cond": TF.to_tensor(cond),
            "gt": TF.to_tensor(gt),
            "prior": TF.to_tensor(prior),
        }
        if m_local is not None:
            result["m_local"] = TF.to_tensor(m_local)
        return result


def load_pipeline(args, device: torch.device, dtype: torch.dtype) -> OneStepPipeline:
    controlnet = ControlNetVAEModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )

    pipe = OneStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    for module in (controlnet, unet, vae, text_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    return pipe


def infer_diff_prelim(
    pipeline: OneStepPipeline,
    image_tensor: torch.Tensor,
    prior_tensor: torch.Tensor,
    prompt: str,
    beta: float,
    m_local_tensor: torch.Tensor | None = None,
    m_local_lambda: float = 0.5,
) -> torch.Tensor:
    device = pipeline._execution_device
    dtype = pipeline.dtype

    if pipeline.empty_text_embedding is None:
        text_inputs = pipeline.tokenizer(
            "",
            padding="do_not_pad",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        pipeline.empty_text_embedding = pipeline.text_encoder(text_input_ids)[0]

    if pipeline.prompt_embeds is None or pipeline.prompt != prompt:
        pipeline.prompt = prompt
        pipeline.prompt_embeds = None

    if pipeline.prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            pipeline.prompt,
            device,
            1,
            False,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        pipeline.prompt_embeds = prompt_embeds
        pipeline.negative_prompt_embeds = negative_prompt_embeds

    image, padding, original_resolution = pipeline.image_processor.preprocess(
        image_tensor, pipeline.default_processing_resolution, "bilinear", device, dtype
    )
    if pipeline.prompt_embeds.shape[0] != image.shape[0]:
        pipeline.prompt_embeds = pipeline.prompt_embeds.repeat(image.shape[0], 1, 1)
    image_latent, pred_latent = pipeline.prepare_latents(image, None, None, 1, 1)

    prior = prior_tensor.to(device=device, dtype=dtype)
    prior = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if m_local_tensor is not None:
        m_local = m_local_tensor.to(device=device, dtype=dtype)
        m_local = F.interpolate(m_local, size=image.shape[-2:], mode="bilinear", align_corners=False)
        prior = (prior + m_local_lambda * m_local).clamp(0.0, 1.0)
    hf_mag = compute_hf_mag(image)

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        image_latent.detach(),
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict=False,
    )

    prior_l = prior.clamp(0.0, 1.0)
    hf_l = hf_mag.clamp(0.0, 1.0)
    gated_down = []
    for res in down_block_res_samples:
        prior_res = F.interpolate(prior_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        hf_res = F.interpolate(hf_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        gate = (1.0 + beta * prior_res * hf_res).clamp(1.0, 1.0 + beta)
        gated_down.append(res * gate)

    prior_mid = F.interpolate(prior_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    hf_mid = F.interpolate(hf_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    gate_mid = (1.0 + beta * prior_mid * hf_mid).clamp(1.0, 1.0 + beta)

    latent_x_t = pipeline.unet(
        pred_latent,
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        down_block_additional_residuals=gated_down,
        mid_block_additional_residual=mid_block_res_sample * gate_mid,
        return_dict=False,
    )[0]

    prediction = pipeline.decode_prediction(latent_x_t)
    prediction = pipeline.image_processor.unpad_image(prediction, padding)
    return pipeline.image_processor.resize_antialias(
        prediction, original_resolution, "bilinear", is_aa=False
    )


def build_refine_input(
    prelim: torch.Tensor,
    cond: torch.Tensor,
    prior: torch.Tensor,
    m_local: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [prelim, compute_hf_image(cond), prior]
    if m_local is not None:
        parts.append(m_local)
    parts.append(cond)
    return torch.cat(parts, dim=1)


def apply_refine_residual(
    refine_net,
    refine_head,
    prelim: torch.Tensor,
    cond: torch.Tensor,
    prior: torch.Tensor,
    m_local: torch.Tensor | None = None,
    residual_scale: float = 0.1,
) -> torch.Tensor:
    feat = refine_net(build_refine_input(prelim, cond, prior, m_local))
    residual = torch.tanh(refine_head(feat)) * residual_scale
    return (prelim + residual).clamp(0.0, 1.0)


def load_refine_models(args, net_path: Path, head_path: Path, device: torch.device):
    in_ch = 10 + int(getattr(args, "use_m_local_refine", False))
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)
    try:
        refine_net.load_state_dict(torch.load(net_path, map_location="cpu"))
        refine_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load refine checkpoint with in_ch={in_ch}. "
            "Check that use_m_local_refine matches the checkpoint (10ch off, 11ch on)."
        ) from exc
    refine_net.eval()
    refine_head.eval()
    return refine_net, refine_head


def find_latest_refine_dir(outputs_dir: Path) -> Path | None:
    candidates = []
    for run_dir in outputs_dir.glob("refine_*"):
        best = run_dir / "best"
        if (best / "nafnet_refine.pth").exists() and (best / "nafnet_refine_head.pth").exists():
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_refine_paths(
    refine_dir: str | Path | None,
    refine_net_path: str | Path | None,
    refine_head_path: str | Path | None,
    refine_config: dict,
    use_final_refine: bool = False,
) -> tuple[Path, Path, Path | None]:
    if refine_net_path and refine_head_path:
        return Path(refine_net_path), Path(refine_head_path), None

    resolved_dir = Path(refine_dir) if refine_dir else None
    if resolved_dir is None:
        output_base = Path(refine_config.get("paths", {}).get("output_dir", "outputs"))
        if output_base.name != "outputs":
            output_base = output_base.parent
        resolved_dir = find_latest_refine_dir(output_base)
        if resolved_dir is None:
            raise FileNotFoundError(
                "Could not auto-detect refine weights. Pass --refine_dir or "
                "--refine_net_path/--refine_head_path."
            )

    if use_final_refine:
        net_path = resolved_dir / "nafnet_refine_final.pth"
        head_path = resolved_dir / "nafnet_refine_head_final.pth"
    else:
        best_dir = resolved_dir if resolved_dir.name == "best" else resolved_dir / "best"
        net_path = best_dir / "nafnet_refine.pth"
        head_path = best_dir / "nafnet_refine_head.pth"

    if not net_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Missing refine weights: {net_path}, {head_path}")
    return net_path, head_path, resolved_dir


def make_pipeline_args(pretrained_model_name_or_path, controlnet_dir, unet_dir):
    return SimpleNamespace(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        controlnet_dir=controlnet_dir,
        unet_dir=unet_dir,
    )
