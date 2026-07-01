#!/usr/bin/env python
# coding=utf-8
"""
Train refinement network with frozen SD (ControlNet+UNet+VAE) and gating.

Inputs come from jsonl entries with fields:
  - conditioning_image: path to input image
  - image: path to GT image
  - prior: path to prior npy

Pipeline:
  cond -> diffusion (ControlNet -> gate -> UNet -> VAE decode) -> prelim (pixel)
  hf  <- wavelet high-frequency of cond
  prior <- prior npy to 1xHxW
  refine: NAFNet(prelim+hf+prior+cond) -> refined
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from basicsr.models.archs.NAFNet_arch import NAFNet
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer

from diffusion.controlnetvae import ControlNetVAEModel
from diffusion.pipeline_onestep import OneStepPipeline
from fumo_mlocal import load_map_pil, resolve_m_local_path
from wavelet_color_fix import wavelet_decomposition


logger = get_logger(__name__)


def parse_args(input_args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refinement network with frozen SD.")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing jsonl files.")
    parser.add_argument(
        "--multiple_datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of jsonl filenames for training.",
    )
    parser.add_argument(
        "--multiple_datasets_probabilities",
        type=float,
        nargs="+",
        required=True,
        help="Sampling probabilities for each jsonl.",
    )
    parser.add_argument("--output_dir", type=str, default="refine_diff_outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--resize_scale", type=float, default=1.1)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--validation_jsonl", type=str, default=None)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--validation_num_workers", type=int, default=None)
    parser.add_argument("--validation_example_index", type=int, default=0)
    parser.add_argument("--validation_resolution_mode", type=str, default="square", choices=["square", "full"])
    parser.add_argument("--validation_resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--validation_num_images", type=int, default=None)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--lpips_weight", type=float, default=0.25)
    parser.add_argument("--grad_weight", type=float, default=0.25)
    parser.add_argument("--nafnet_width", type=int, default=64)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=1)
    parser.add_argument(
        "--nafnet_enc_blk_nums",
        type=int,
        nargs="+",
        default=[1, 1, 1, 28],
        help="Encoder block numbers for NAFNet, e.g. --nafnet_enc_blk_nums 1 1 1 28",
    )
    parser.add_argument(
        "--nafnet_dec_blk_nums",
        type=int,
        nargs="+",
        default=[1, 1, 1, 1],
        help="Decoder block numbers for NAFNet, e.g. --nafnet_dec_blk_nums 1 1 1 1",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        "--model_id",
        type=str,
        default="./HF_CACHE/weights",
        help="Base model id/path for VAE, tokenizer and text encoder.",
    )
    parser.add_argument("--controlnet_dir", type=str, required=True)
    parser.add_argument("--unet_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="remove glass reflection")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--use_m_local_diffusion", action="store_true", help="Use M_local in the diffusion prelim gate.")
    parser.add_argument("--use_m_local_refine", action="store_true", help="Concatenate M_local into the refine network input.")
    parser.add_argument("--m_local_dirs", type=str, nargs="*", default=None, help="Directories searched by conditioning image stem for M_local npy files.")
    parser.add_argument("--m_local_column", type=str, default="m_local", help="Jsonl column containing an optional M_local npy path.")
    parser.add_argument("--m_local_lambda", type=float, default=0.5, help="Weight applied to M_local in the diffusion gate prior.")
    parser.add_argument("--m_local_missing_policy", type=str, default="error", choices=["error", "zero"], help="How to handle missing M_local maps when M_local is enabled.")
    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def load_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _normalize_to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_hf_image(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf = _normalize_to_01(high_freq)
    return hf


def compute_hf_mag(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    hf_mag = (hf_mag / mean).clamp(0.0, 1.0)
    return hf_mag


def load_prior_tensor(prior_path: str) -> torch.Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    return TF.to_tensor(prior_img)  # [1,H,W]


FINAL_SCORE_SSIM_WEIGHT = 10.0
FINAL_SCORE_LPIPS_WEIGHT = 5.0


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


def create_training_history() -> dict:
    return {"train": [], "validation": []}


def update_training_history(history: dict, split: str, step: int, values: dict) -> None:
    payload = {"step": int(step)}
    payload.update({key: float(value) for key, value in values.items() if value is not None})
    history.setdefault(split, []).append(payload)


def _plot_metric(axis, history: dict, metric: str) -> bool:
    has_data = False
    for split, entries in history.items():
        xs = [entry["step"] for entry in entries if metric in entry]
        ys = [entry[metric] for entry in entries if metric in entry]
        if xs:
            axis.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=split)
            has_data = True
    return has_data


def save_training_history(history: dict, output_dir: str) -> None:
    log_dir = Path(output_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping history plot because matplotlib is unavailable: %s", exc)
        return

    metrics = [
        "loss/total",
        "loss/l1",
        "loss/lpips",
        "loss/grad",
        "lr",
        "psnr",
        "ssim",
        "lpips",
        "final_score",
    ]
    for metric in metrics:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        has_data = _plot_metric(axis, history, metric)
        axis.set_title(metric)
        axis.set_xlabel("step")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        if has_data:
            axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(log_dir / f"{metric.replace('/', '_')}.png", dpi=160)
        plt.close(fig)


def setup_rank0_file_logging(output_dir: str) -> None:
    log_dir = Path(output_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root_logger.addHandler(file_handler)
    base_logger = getattr(logger, "logger", None)
    if base_logger is not None:
        base_logger.addHandler(file_handler)


def unwrap_for_state_dict(module, accelerator: Accelerator | None = None):
    return accelerator.unwrap_model(module) if accelerator is not None else module


def save_refine_modules(
    refine_net,
    refine_head,
    output_dir: str | Path,
    net_name: str = "nafnet_refine.pth",
    head_name: str = "nafnet_refine_head.pth",
    accelerator: Accelerator | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_for_state_dict(refine_net, accelerator).state_dict(), output_dir / net_name)
    torch.save(unwrap_for_state_dict(refine_head, accelerator).state_dict(), output_dir / head_name)


def save_refine_best_model(refine_net, refine_head, accelerator, output_dir: str, step: int, metrics: dict) -> None:
    best_dir = Path(output_dir) / "best"
    save_refine_modules(refine_net, refine_head, best_dir, accelerator=accelerator)
    payload = {"step": int(step), **{key: float(value) for key, value in metrics.items()}}
    with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def clone_ema_module(module):
    ema_module = copy.deepcopy(module)
    ema_module.eval()
    for parameter in ema_module.parameters():
        parameter.requires_grad_(False)
    return ema_module


@torch.no_grad()
def copy_module_state(target, source) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def update_ema_module(ema_module, source_module, decay: float) -> None:
    ema_state = ema_module.state_dict()
    source_state = source_module.state_dict()
    for key, ema_value in ema_state.items():
        source_value = source_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source_value.to(device=ema_value.device, dtype=ema_value.dtype), alpha=1.0 - decay)
        else:
            ema_value.copy_(source_value.to(device=ema_value.device))


def save_ema_checkpoint(ema_refine_net, ema_refine_head, checkpoint_dir: str | Path) -> None:
    save_refine_modules(
        ema_refine_net,
        ema_refine_head,
        checkpoint_dir,
        net_name="ema_nafnet_refine.pth",
        head_name="ema_nafnet_refine_head.pth",
        accelerator=None,
    )


def load_ema_checkpoint(ema_refine_net, ema_refine_head, checkpoint_dir: str | Path, device: torch.device) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    net_path = checkpoint_dir / "ema_nafnet_refine.pth"
    head_path = checkpoint_dir / "ema_nafnet_refine_head.pth"
    if not net_path.exists() or not head_path.exists():
        return False
    ema_refine_net.load_state_dict(torch.load(net_path, map_location=device))
    ema_refine_head.load_state_dict(torch.load(head_path, map_location=device))
    return True


class JsonlDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        resolution: int,
        resize_scale: float,
        disable_augment: bool,
        load_m_local: bool = False,
        m_local_dirs: List[str] | None = None,
        m_local_column: str = "m_local",
        m_local_missing_policy: str = "error",
    ):
        self.entries = entries
        self.resolution = resolution
        self.resize_scale = resize_scale
        self.disable_augment = disable_augment
        self.load_m_local = load_m_local
        self.m_local_dirs = m_local_dirs
        self.m_local_column = m_local_column
        self.m_local_missing_policy = m_local_missing_policy

    def __len__(self) -> int:
        return len(self.entries)

    def _resize(self, img: Image.Image, size: int) -> Image.Image:
        return img.resize((size, size), Image.Resampling.BILINEAR)

    def __getitem__(self, idx: int) -> dict:
        item = self.entries[idx]
        cond_path = item["conditioning_image"]
        gt_path = item["image"]
        prior_path = item["prior"]

        cond = Image.open(cond_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        prior = load_prior_tensor(prior_path)
        prior = TF.to_pil_image(prior)
        if self.load_m_local:
            resolved_m_local_path = resolve_m_local_path(
                item,
                cond_path,
                self.m_local_dirs,
                self.m_local_column,
                self.m_local_missing_policy,
            )
            m_local = Image.new("L", cond.size, color=0) if resolved_m_local_path is None else load_map_pil(resolved_m_local_path)
        else:
            m_local = None

        if self.disable_augment:
            cond = self._resize(cond, self.resolution)
            gt = self._resize(gt, self.resolution)
            prior = self._resize(prior, self.resolution)
            if m_local is not None:
                m_local = self._resize(m_local, self.resolution)
            if prior.size != cond.size:
                prior = prior.resize(cond.size, Image.Resampling.LANCZOS)
            if m_local is not None and m_local.size != cond.size:
                m_local = m_local.resize(cond.size, Image.Resampling.LANCZOS)
        else:
            resize_size = int(self.resolution * self.resize_scale)
            cond = self._resize(cond, resize_size)
            gt = self._resize(gt, resize_size)
            prior = self._resize(prior, self.resolution)
            if m_local is not None:
                m_local = self._resize(m_local, self.resolution)
            if prior.size != cond.size:
                prior = prior.resize(cond.size, Image.Resampling.LANCZOS)
            if m_local is not None and m_local.size != cond.size:
                m_local = m_local.resize(cond.size, Image.Resampling.LANCZOS)

            i, j = torch.randint(0, resize_size - self.resolution + 1, (2,)).tolist()
            cond = TF.crop(cond, i, j, self.resolution, self.resolution)
            gt = TF.crop(gt, i, j, self.resolution, self.resolution)
            prior = TF.crop(prior, i, j, self.resolution, self.resolution)
            if m_local is not None:
                m_local = TF.crop(m_local, i, j, self.resolution, self.resolution)

            if random.random() < 0.5:
                cond = TF.hflip(cond)
                gt = TF.hflip(gt)
                prior = TF.hflip(prior)
                if m_local is not None:
                    m_local = TF.hflip(m_local)
            if random.random() < 0.5:
                cond = TF.vflip(cond)
                gt = TF.vflip(gt)
                prior = TF.vflip(prior)
                if m_local is not None:
                    m_local = TF.vflip(m_local)

            rotate_k = random.randint(0, 3)
            if rotate_k:
                angle = 90 * rotate_k
                cond = cond.rotate(angle, expand=False)
                gt = gt.rotate(angle, expand=False)
                prior = prior.rotate(angle, expand=False)
                if m_local is not None:
                    m_local = m_local.rotate(angle, expand=False)

        cond = TF.to_tensor(cond)  # [0,1]
        gt = TF.to_tensor(gt)      # [0,1]
        prior = TF.to_tensor(prior)  # [1,H,W]
        result = {"cond": cond, "gt": gt, "prior": prior}
        if m_local is not None:
            result["m_local"] = TF.to_tensor(m_local)

        if prior.shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"Prior size {prior.shape[-2:]} does not match cond size {cond.shape[-2:]}."
            )
        if "m_local" in result and result["m_local"].shape[-2:] != cond.shape[-2:]:
            raise ValueError(
                f"M_local size {result['m_local'].shape[-2:]} does not match cond size {cond.shape[-2:]}."
            )

        return result


class ValidationJsonlDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        resolution: int,
        resolution_mode: str = "square",
        resize: Tuple[int, int] | None = None,
        num_images: int | None = None,
        load_m_local: bool = False,
        m_local_dirs: List[str] | None = None,
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


class FuseDataset(Dataset):
    def __init__(self, datasets: List[Dataset], probabilities: List[float]):
        self.datasets = datasets
        self.cum_probs = np.cumsum(probabilities)

    def __len__(self) -> int:
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> dict:
        r = random.random()
        dataset_idx = int(np.searchsorted(self.cum_probs, r))
        dataset = self.datasets[dataset_idx]
        return dataset[idx % len(dataset)]


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> OneStepPipeline:
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

    for p in controlnet.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    controlnet.eval()
    unet.eval()
    vae.eval()
    text_encoder.eval()
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

    gated_down = []
    prior_l = prior.clamp(0.0, 1.0)
    hf_l = hf_mag.clamp(0.0, 1.0)
    for res in down_block_res_samples:
        prior_res = F.interpolate(prior_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        hf_res = F.interpolate(hf_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        gate = 1.0 + beta * prior_res * hf_res
        gate = gate.clamp(1.0, 1.0 + beta)
        gated_down.append(res * gate)
    down_block_res_samples = gated_down

    prior_mid = F.interpolate(prior_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    hf_mid = F.interpolate(hf_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    gate_mid = 1.0 + beta * prior_mid * hf_mid
    gate_mid = gate_mid.clamp(1.0, 1.0 + beta)
    mid_block_res_sample = mid_block_res_sample * gate_mid

    latent_x_t = pipeline.unet(
        pred_latent,
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]

    prediction = pipeline.decode_prediction(latent_x_t)
    prediction = pipeline.image_processor.unpad_image(prediction, padding)
    prediction = pipeline.image_processor.resize_antialias(
        prediction, original_resolution, "bilinear", is_aa=False
    )
    return prediction


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - gt))


def normalize_tensor_for_lpips(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 2.0) - 1.0


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    device = pred.device
    dtype = pred.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)

    channels = pred.shape[1]
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)

    pred_dx = F.conv2d(pred, kx, padding=1, groups=channels)
    pred_dy = F.conv2d(pred, ky, padding=1, groups=channels)
    gt_dx = F.conv2d(gt, kx, padding=1, groups=channels)
    gt_dy = F.conv2d(gt, ky, padding=1, groups=channels)

    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def build_refine_input(
    prelim: torch.Tensor,
    cond: torch.Tensor,
    prior: torch.Tensor,
    m_local: torch.Tensor | None = None,
) -> torch.Tensor:
    hf = compute_hf_image(cond)
    parts = [prelim, hf, prior]
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


def log_validation(
    pipeline: OneStepPipeline,
    refine_net,
    refine_head,
    val_dataloader: DataLoader,
    args: argparse.Namespace,
    accelerator: Accelerator,
    lpips_metric,
    step: int,
) -> dict:
    logger.info(
        "Running refine validation... mode=%s resize=%s num_images=%s",
        args.validation_resolution_mode,
        args.validation_resize,
        args.validation_num_images,
    )
    refine_net_was_training = refine_net.training
    refine_head_was_training = refine_head.training
    refine_net.eval()
    refine_head.eval()

    metric_sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "final_score": 0.0}
    count = 0
    image_log = None
    save_idx = max(args.validation_example_index, 0)
    seen = 0

    val_iter = tqdm(
        val_dataloader,
        desc=f"Refine validation step {step}",
        disable=not accelerator.is_main_process,
        leave=True,
    )

    with torch.no_grad():
        for batch in val_iter:
            cond = batch["cond"].to(accelerator.device)
            gt = batch["gt"].to(accelerator.device)
            prior = batch["prior"].to(accelerator.device)
            m_local = batch.get("m_local")
            if m_local is not None:
                m_local = m_local.to(accelerator.device)

            prelim_pred = infer_diff_prelim(
                pipeline,
                cond,
                prior,
                args.prompt,
                args.beta,
                m_local_tensor=m_local if args.use_m_local_diffusion else None,
                m_local_lambda=args.m_local_lambda,
            )
            prelim = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0)
            refined = apply_refine_residual(
                refine_net,
                refine_head,
                prelim,
                cond,
                prior,
                m_local=m_local if args.use_m_local_refine else None,
            )

            metrics = calculate_validation_metrics(refined, gt, lpips_metric)
            batch_size = cond.shape[0]
            for key, value in metrics.items():
                metric_sums[key] += value * batch_size
            count += batch_size

            if image_log is None and seen <= save_idx < seen + batch_size:
                local_idx = save_idx - seen
                image_log = {
                    "cond": tensor_to_pil(cond[local_idx]),
                    "prelim": tensor_to_pil(prelim[local_idx]),
                    "refined": tensor_to_pil(refined[local_idx]),
                    "gt": tensor_to_pil(gt[local_idx]),
                }
            seen += batch_size

    if count == 0:
        raise ValueError("Validation dataloader is empty.")

    avg_metrics = {key: value / count for key, value in metric_sums.items()}
    logger.info(
        "Refine Validation Metrics [Avg] - PSNR: %.4f, SSIM: %.4f, LPIPS: %.4f, Final: %.4f",
        avg_metrics["psnr"],
        avg_metrics["ssim"],
        avg_metrics["lpips"],
        avg_metrics["final_score"],
    )

    refine_net.train(refine_net_was_training)
    refine_head.train(refine_head_was_training)
    return {"metrics": avg_metrics, "image_log": image_log}


def save_validation_example(image_log: dict, output_dir: str, step: int) -> None:
    if not image_log:
        return
    image_dir = Path(output_dir) / "validation_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    grid = labeled_image_grid(
        [image_log["cond"], image_log["prelim"], image_log["refined"], image_log["gt"]],
        ["LQ", "prelim", "prediction", "GT"],
    )
    grid.save(image_dir / f"step_{int(step):06d}.png")


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if len(args.nafnet_enc_blk_nums) != len(args.nafnet_dec_blk_nums):
        raise ValueError("nafnet_enc_blk_nums and nafnet_dec_blk_nums must have the same length.")
    if args.nafnet_middle_blk_num < 0:
        raise ValueError("nafnet_middle_blk_num must be >= 0.")

    if args.report_to is not None and args.report_to.lower() in {"none", "null", "false"}:
        args.report_to = None

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        setup_rank0_file_logging(args.output_dir)
        tracker_config = dict(vars(args))
        tracker_config["nafnet_enc_blk_nums"] = ",".join(map(str, args.nafnet_enc_blk_nums))
        tracker_config["nafnet_dec_blk_nums"] = ",".join(map(str, args.nafnet_dec_blk_nums))
        if tracker_config.get("m_local_dirs") is not None:
            tracker_config["m_local_dirs"] = ",".join(map(str, tracker_config["m_local_dirs"]))
        tracker_config.pop("multiple_datasets", None)
        tracker_config.pop("multiple_datasets_probabilities", None)
        if args.report_to is not None:
            accelerator.init_trackers("refine_diff", config=tracker_config)

    if len(args.multiple_datasets) != len(args.multiple_datasets_probabilities):
        raise ValueError("multiple_datasets and multiple_datasets_probabilities must have same length.")

    probabilities = np.array(args.multiple_datasets_probabilities, dtype=np.float32)
    probabilities = probabilities / probabilities.sum()

    datasets = []
    for name in args.multiple_datasets:
        jsonl_path = os.path.join(args.train_data_dir, name)
        entries = load_jsonl(jsonl_path)
        datasets.append(
            JsonlDataset(
                entries,
                args.resolution,
                args.resize_scale,
                args.disable_augment,
                load_m_local=args.use_m_local_diffusion or args.use_m_local_refine,
                m_local_dirs=args.m_local_dirs,
                m_local_column=args.m_local_column,
                m_local_missing_policy=args.m_local_missing_policy,
            )
        )

    train_dataset = FuseDataset(datasets, probabilities.tolist())
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = None
    if args.validation_jsonl is not None:
        val_entries = load_jsonl(args.validation_jsonl)
        val_dataset = ValidationJsonlDataset(
            val_entries,
            resolution=args.resolution,
            resolution_mode=args.validation_resolution_mode,
            resize=args.validation_resize,
            num_images=args.validation_num_images,
            load_m_local=args.use_m_local_diffusion or args.use_m_local_refine,
            m_local_dirs=args.m_local_dirs,
            m_local_column=args.m_local_column,
            m_local_missing_policy=args.m_local_missing_policy,
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.validation_batch_size,
            num_workers=args.validation_num_workers if args.validation_num_workers is not None else args.num_workers,
            pin_memory=True,
        )

    device = accelerator.device
    dtype = torch.float32

    pipeline = load_pipeline(args, device, dtype)

    in_ch = 10 + int(args.use_m_local_refine)
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_net.train()

    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)
    torch.nn.init.zeros_(refine_head.weight)
    torch.nn.init.zeros_(refine_head.bias)
    refine_head.train()

    ema_refine_net = None
    ema_refine_head = None
    if args.use_ema:
        ema_refine_net = clone_ema_module(refine_net)
        ema_refine_head = clone_ema_module(refine_head)
        logger.info("Using refine EMA with decay %.6f", args.ema_decay)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    validation_lpips_model = lpips.LPIPS(net="alex").to(device)
    validation_lpips_model.eval()
    for p in validation_lpips_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(refine_net.parameters()) + list(refine_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def get_cosine_scheduler(optimizer, max_steps: int, warmup_steps: int, min_lr: float):
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 for cosine scheduler.")

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return (min_lr / args.learning_rate) + (1.0 - min_lr / args.learning_rate) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    refine_net, refine_head, optimizer, train_dataloader = accelerator.prepare(
        refine_net, refine_head, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_scheduler(
        optimizer,
        max_steps=max_train_steps,
        warmup_steps=args.lr_warmup_steps,
        min_lr=args.min_learning_rate,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Max train steps = {max_train_steps}")

    resume_step = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(args.output_dir) / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint_path}")
        accelerator.print(f"Resuming refine training from checkpoint: {checkpoint_path}")
        accelerator.load_state(str(checkpoint_path))
        if args.use_ema:
            if load_ema_checkpoint(ema_refine_net, ema_refine_head, checkpoint_path, device):
                logger.info("Loaded refine EMA weights from %s", checkpoint_path)
            else:
                logger.info("No EMA weights found in %s; initializing EMA from resumed refine weights", checkpoint_path)
                copy_module_state(ema_refine_net, accelerator.unwrap_model(refine_net))
                copy_module_state(ema_refine_head, accelerator.unwrap_model(refine_head))
        try:
            resume_step = int(checkpoint_path.name.split("-")[-1])
        except ValueError:
            resume_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=resume_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = resume_step
    if resume_step > 0:
        lr_scheduler.last_epoch = resume_step

    best_validation_score = float("-inf")
    best_metrics_path = Path(args.output_dir) / "best" / "metrics.json"
    if best_metrics_path.exists():
        try:
            best_validation_score = float(json.loads(best_metrics_path.read_text(encoding="utf-8"))["final_score"])
            logger.info("Loaded existing best refine final_score: %.4f", best_validation_score)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to load existing best metrics from %s", best_metrics_path)
    history = create_training_history()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(refine_net):
                cond = batch["cond"].to(device)
                gt = batch["gt"].to(device)
                prior = batch["prior"].to(device)
                m_local = batch.get("m_local")
                if m_local is not None:
                    m_local = m_local.to(device)

                with torch.no_grad():
                    prelim_pred = infer_diff_prelim(
                        pipeline,
                        cond,
                        prior,
                        args.prompt,
                        args.beta,
                        m_local_tensor=m_local if args.use_m_local_diffusion else None,
                        m_local_lambda=args.m_local_lambda,
                    )
                    prelim = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0)

                refined = apply_refine_residual(
                    refine_net,
                    refine_head,
                    prelim,
                    cond,
                    prior,
                    m_local=m_local if args.use_m_local_refine else None,
                )

                l1_val = l1_loss(refined, gt)
                lpips_val = lpips_model(
                    normalize_tensor_for_lpips(refined),
                    normalize_tensor_for_lpips(gt),
                ).mean()
                grad_val = gradient_loss(refined, gt)
                loss = (
                    args.l1_weight * l1_val
                    + args.lpips_weight * lpips_val
                    + args.grad_weight * grad_val
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                lr_scheduler.step()
                if args.use_ema:
                    update_ema_module(ema_refine_net, accelerator.unwrap_model(refine_net), args.ema_decay)
                    update_ema_module(ema_refine_head, accelerator.unwrap_model(refine_head), args.ema_decay)
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.save_state(ckpt_dir)
                    if args.use_ema:
                        save_ema_checkpoint(ema_refine_net, ema_refine_head, ckpt_dir)

                    if args.checkpoints_total_limit is not None:
                        checkpoints = [
                            d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")
                        ]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) > args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                            for removing in checkpoints[:num_to_remove]:
                                removing_path = os.path.join(args.output_dir, removing)
                                try:
                                    shutil.rmtree(removing_path)
                                except OSError:
                                    pass

                if (
                    accelerator.is_main_process
                    and val_dataloader is not None
                    and global_step % args.validation_steps == 0
                ):
                    validation_refine_net = ema_refine_net if args.use_ema else accelerator.unwrap_model(refine_net)
                    validation_refine_head = ema_refine_head if args.use_ema else accelerator.unwrap_model(refine_head)
                    validation_output = log_validation(
                        pipeline,
                        validation_refine_net,
                        validation_refine_head,
                        val_dataloader,
                        args,
                        accelerator,
                        validation_lpips_model,
                        global_step,
                    )
                    metrics = validation_output["metrics"]
                    save_validation_example(validation_output["image_log"], args.output_dir, global_step)
                    update_training_history(history, "validation", global_step, metrics)
                    save_training_history(history, args.output_dir)
                    if metrics["final_score"] > best_validation_score:
                        best_validation_score = metrics["final_score"]
                        logger.info(
                            "New best refine final_score: %.4f. Saving best refine modules...",
                            best_validation_score,
                        )
                        best_refine_net = ema_refine_net if args.use_ema else refine_net
                        best_refine_head = ema_refine_head if args.use_ema else refine_head
                        best_accelerator = None if args.use_ema else accelerator
                        save_refine_best_model(best_refine_net, best_refine_head, best_accelerator, args.output_dir, global_step, metrics)

                if val_dataloader is not None and global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()

            logs = {
                "loss/total": loss.detach().item(),
                "loss/l1": l1_val.detach().item(),
                "loss/lpips": lpips_val.detach().item(),
                "loss/grad": grad_val.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(loss=logs["loss/total"])
            if accelerator.is_main_process and global_step % args.log_interval == 0:
                update_training_history(history, "train", global_step, logs)
                save_training_history(history, args.output_dir)
                logger.info(
                    "Iteration [%d/%d] Loss: %.6f L1: %.6f LPIPS: %.6f Grad: %.6f LR: %.8f",
                    global_step,
                    max_train_steps,
                    logs["loss/total"],
                    logs["loss/l1"],
                    logs["loss/lpips"],
                    logs["loss/grad"],
                    logs["lr"],
                )
            accelerator.log(
                {
                    "loss/total": logs["loss/total"],
                    "loss/l1": logs["loss/l1"],
                    "loss/lpips": logs["loss/lpips"],
                    "loss/grad": logs["loss/grad"],
                    "lr": logs["lr"],
                },
                step=global_step,
            )

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_refine_net = ema_refine_net if args.use_ema else refine_net
        final_refine_head = ema_refine_head if args.use_ema else refine_head
        final_accelerator = None if args.use_ema else accelerator
        save_refine_modules(
            final_refine_net,
            final_refine_head,
            args.output_dir,
            net_name="nafnet_refine_final.pth",
            head_name="nafnet_refine_head_final.pth",
            accelerator=final_accelerator,
        )
        if args.use_ema:
            save_refine_modules(
                refine_net,
                refine_head,
                args.output_dir,
                net_name="nafnet_refine_raw_final.pth",
                head_name="nafnet_refine_head_raw_final.pth",
                accelerator=accelerator,
            )

        if val_dataloader is not None:
            validation_refine_net = ema_refine_net if args.use_ema else accelerator.unwrap_model(refine_net)
            validation_refine_head = ema_refine_head if args.use_ema else accelerator.unwrap_model(refine_head)
            validation_output = log_validation(
                pipeline,
                validation_refine_net,
                validation_refine_head,
                val_dataloader,
                args,
                accelerator,
                validation_lpips_model,
                global_step,
            )
            metrics = validation_output["metrics"]
            save_validation_example(validation_output["image_log"], args.output_dir, global_step)
            update_training_history(history, "validation", global_step, metrics)
            save_training_history(history, args.output_dir)
            if metrics["final_score"] > best_validation_score:
                best_refine_net = ema_refine_net if args.use_ema else refine_net
                best_refine_head = ema_refine_head if args.use_ema else refine_head
                best_accelerator = None if args.use_ema else accelerator
                save_refine_best_model(best_refine_net, best_refine_head, best_accelerator, args.output_dir, global_step, metrics)


if __name__ == "__main__":
    main()
