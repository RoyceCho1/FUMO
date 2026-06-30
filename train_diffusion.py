#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import argparse
import contextlib
import copy
import gc
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusion.controlnetvae import ControlNetVAEModel

from diffusion.pipeline_onestep import OneStepPipeline
from fumo_mlocal import load_map_array, load_map_pil, normalize_map_array, resolve_m_local_path
from wavelet_color_fix import wavelet_decomposition



if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def labeled_image_grid(imgs, labels):
    assert len(imgs) == len(labels)

    w, h = imgs[0].size
    label_height = max(36, h // 18)
    grid = Image.new("RGB", size=(len(imgs) * w, h + label_height), color="white")

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i * w, 0))

    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, w // 36))
    except OSError:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = i * w + (w - text_w) // 2
        y = h + (label_height - text_h) // 2
        draw.text((x, y), label, fill="black", font=font)

    return grid


def compute_beta(global_step: int, max_steps: int, warmup_ratio: float, beta_max: float) -> float:
    if max_steps <= 0:
        return beta_max
    warmup_steps = int(max_steps * warmup_ratio)
    if global_step < warmup_steps:
        return 0.0
    progress = (global_step - warmup_steps) / max(1, max_steps - warmup_steps)
    return beta_max * min(max(progress, 0.0), 1.0)


def compute_hf_mag(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    hf_mag = (hf_mag / mean).clamp(0.0, 1.0)
    return hf_mag


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


def build_lpips_metric(device: torch.device):
    import lpips

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


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


def unwrap_for_save(model, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def save_stage1_modules(controlnet, unet, output_dir: str | Path, accelerator=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrap_for_save(controlnet, accelerator).save_pretrained(output_dir / "controlnet")
    unwrap_for_save(unet, accelerator).save_pretrained(output_dir / "unet")


def save_stage1_best_model(controlnet, unet, accelerator, output_dir: str, step: int, metrics: dict) -> None:
    best_dir = Path(output_dir) / "best"
    save_stage1_modules(controlnet, unet, best_dir, accelerator=accelerator)
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


def save_diffusion_ema_checkpoint(ema_controlnet, ema_unet, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if ema_controlnet is not None:
        ema_controlnet.save_pretrained(checkpoint_dir / "controlnet_ema")
    if ema_unet is not None:
        ema_unet.save_pretrained(checkpoint_dir / "unet_ema")


def load_diffusion_ema_checkpoint(ema_controlnet, ema_unet, checkpoint_dir: str | Path, torch_dtype=None) -> tuple[bool, bool]:
    checkpoint_dir = Path(checkpoint_dir)
    loaded_controlnet = False
    loaded_unet = False
    if ema_controlnet is not None and (checkpoint_dir / "controlnet_ema").exists():
        loaded = ControlNetVAEModel.from_pretrained(checkpoint_dir / "controlnet_ema", torch_dtype=torch_dtype)
        ema_controlnet.load_state_dict(loaded.state_dict())
        del loaded
        loaded_controlnet = True
    if ema_unet is not None and (checkpoint_dir / "unet_ema").exists():
        loaded = UNet2DConditionModel.from_pretrained(checkpoint_dir / "unet_ema", torch_dtype=torch_dtype)
        ema_unet.load_state_dict(loaded.state_dict())
        del loaded
        loaded_unet = True
    return loaded_controlnet, loaded_unet


def create_training_history() -> dict:
    return {"train": [], "validation": []}


def update_training_history(history: dict, split: str, step: int, values: dict) -> None:
    payload = {"step": int(step)}
    payload.update({key: float(value) for key, value in values.items() if value is not None})
    history.setdefault(split, []).append(payload)


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

    metrics = ["loss", "psnr", "ssim", "lpips", "final_score"]
    for metric in metrics:
        metric_fig, metric_axis = plt.subplots(figsize=(8, 4.5))
        metric_has_data = _plot_metric(metric_axis, history, metric)
        metric_axis.set_title(metric)
        metric_axis.set_xlabel("step")
        metric_axis.set_ylabel(metric)
        metric_axis.grid(True, alpha=0.3)
        if metric_has_data:
            metric_axis.legend(loc="best")
        metric_fig.tight_layout()
        metric_fig.savefig(log_dir / f"{metric}.png", dpi=160)
        plt.close(metric_fig)


def _plot_metric(axis, history: dict, metric: str) -> bool:
    has_data = False
    for split, entries in history.items():
        xs = [entry["step"] for entry in entries if metric in entry]
        ys = [entry[metric] for entry in entries if metric in entry]
        if xs:
            axis.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=split)
            has_data = True
    return has_data


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


def save_validation_examples(image_logs: list, output_dir: str, step: int) -> None:
    if not image_logs:
        return
    image_dir = Path(output_dir) / "validation_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    log = image_logs[0]
    images = [log["validation_image"]] + log["images"] + [log["gt_image"]]
    labels = ["LQ"] + ["prediction" if len(log["images"]) == 1 else f"prediction {idx + 1}" for idx in range(len(log["images"]))] + ["GT"]
    grid = labeled_image_grid(images, labels)
    grid.save(image_dir / f"step_{int(step):06d}.png")


def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if args.validation_jsonl is None:
        logger.warning("Skipping validation because --validation_jsonl is not set.")
        return {"image_logs": [], "metrics": None}

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
        unet = accelerator.unwrap_model(unet)
    else:
        controlnet = ControlNetVAEModel.from_pretrained(
            os.path.join(args.output_dir, "controlnet"), torch_dtype=weight_dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(args.output_dir, "unet"), torch_dtype=weight_dtype
        )

    if is_final_validation:
        controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)
        unet = unet.to(accelerator.device, dtype=weight_dtype)

    pipeline = OneStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(accelerator.device)

    from torchvision.transforms import ToTensor

    metric_sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "final_score": 0.0}
    count = 0
    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    lpips_metric = build_lpips_metric(accelerator.device)
    beta = compute_beta(step, args.max_train_steps, args.beta_warmup_ratio, args.beta_max)

    with open(args.validation_jsonl, "r", encoding="utf-8") as f:
        validation_data = [json.loads(line) for line in f if line.strip()]

    if not validation_data:
        raise ValueError(f"Validation jsonl is empty: {args.validation_jsonl}")

    save_idx = min(max(args.validation_example_index, 0), len(validation_data) - 1)
    if save_idx != args.validation_example_index:
        logger.warning(
            "validation_example_index %d is out of range for %d rows. Using %d instead.",
            args.validation_example_index,
            len(validation_data),
            save_idx,
        )

    validation_iter = tqdm(
        validation_data,
        desc=f"Validation step {step}",
        disable=not accelerator.is_main_process,
        leave=True,
    )

    for row_idx, data in enumerate(validation_iter):
        validation_image_path = data["conditioning_image"]
        validation_prompt = data.get("text", "remove degradation")
        gt_image_path = data["image"]
        prior_path = data.get("prior")

        validation_image = Image.open(validation_image_path).convert("RGB")
        gt_image = Image.open(gt_image_path).convert("RGB")
        prior = normalize_map_array(np.load(prior_path)) if prior_path else None
        if args.use_m_local_diffusion and prior is not None:
            resolved_m_local_path = resolve_m_local_path(
                data,
                validation_image_path,
                args.m_local_dirs,
                args.m_local_column,
                args.m_local_missing_policy,
            )
            if resolved_m_local_path is None:
                m_local = np.zeros_like(prior, dtype=np.float32)
            else:
                m_local = load_map_array(resolved_m_local_path)
                if m_local.shape != prior.shape:
                    m_local_img = Image.fromarray((m_local * 255.0).astype(np.uint8), mode="L")
                    m_local_img = m_local_img.resize((prior.shape[1], prior.shape[0]), Image.Resampling.LANCZOS)
                    m_local = np.asarray(m_local_img, dtype=np.float32) / 255.0
            prior = np.clip(prior + args.m_local_lambda * m_local, 0.0, 1.0)

        images = []
        with inference_ctx:
            for _ in range(args.num_validation_images):
                generated_image = pipeline(
                    image=validation_image,
                    prompt=validation_prompt,
                    prior=prior,
                    beta=beta,
                    processing_resolution=args.resolution,
                ).prediction[0]

                generated_image = (generated_image + 1) / 2
                generated_image = generated_image.clip(0.0, 1.0)
                generated_image = (generated_image * 255).astype(np.uint8)
                generated_image = Image.fromarray(generated_image)
                images.append(generated_image)

                gen_tensor = ToTensor()(generated_image).unsqueeze(0).to(accelerator.device)
                gt_tensor = ToTensor()(gt_image).unsqueeze(0).to(accelerator.device)

                if gen_tensor.shape[-2:] != gt_tensor.shape[-2:]:
                    gen_tensor = torch.nn.functional.interpolate(
                        gen_tensor,
                        size=gt_tensor.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                metrics = calculate_validation_metrics(gen_tensor, gt_tensor, lpips_metric)
                for key, value in metrics.items():
                    metric_sums[key] += value
                count += 1

        if row_idx == save_idx:
            image_logs.append({
                "validation_image": validation_image,
                "images": images,
                "gt_image": gt_image,
                "validation_prompt": validation_prompt,
            })

    avg_metrics = {key: value / count for key, value in metric_sums.items()}
    logger.info(
        "Validation Metrics [Avg] - PSNR: %.4f, SSIM: %.4f, LPIPS: %.4f, Final: %.4f",
        avg_metrics["psnr"],
        avg_metrics["ssim"],
        avg_metrics["lpips"],
        avg_metrics["final_score"],
    )

    accelerator.log({
        "validation/psnr": avg_metrics["psnr"],
        "validation/ssim": avg_metrics["ssim"],
        "validation/lpips": avg_metrics["lpips"],
        "validation/final_score": avg_metrics["final_score"],
    }, step=step)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                gt_image = log["gt_image"]

                formatted_images = [np.asarray(validation_image)]
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images.append(np.asarray(gt_image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
                for image in images:
                    formatted_images.append(wandb.Image(image, caption=validation_prompt))
            tracker.log({tracker_key: formatted_images})

    del pipeline, lpips_metric
    gc.collect()
    torch.cuda.empty_cache()

    return {"image_logs": image_logs, "metrics": avg_metrics}


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--use_controlnet_ema", action="store_true", help="Maintain EMA weights for ControlNet and use them for validation/best/final saving.")
    parser.add_argument("--use_unet_ema", action="store_true", help="Maintain EMA weights for UNet and use them for validation/best/final saving.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for diffusion modules.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Save train loss history and plot every X optimizer steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--prior_column",
        type=str,
        default="prior",
        help="The column of the dataset containing a prior npy path.",
    )
    parser.add_argument(
        "--m_local_column",
        type=str,
        default="m_local",
        help="The column of the dataset containing an optional M_local npy path.",
    )
    parser.add_argument(
        "--m_local_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Directories searched by conditioning image stem for M_local npy files.",
    )
    parser.add_argument(
        "--use_m_local_diffusion",
        action="store_true",
        help="Use M_local to build the effective diffusion gate prior.",
    )
    parser.add_argument(
        "--m_local_lambda",
        type=float,
        default=0.5,
        help="Weight applied to M_local before adding it to the intensity prior.",
    )
    parser.add_argument(
        "--m_local_missing_policy",
        type=str,
        default="error",
        choices=["error", "zero"],
        help="How to handle missing M_local maps when M_local is enabled.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prior_scale",
        type=float,
        default=0.2,
        help="Scale for modulating conditioning latents with the prior map.",
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        default=0.25,
        help="Maximum beta for gating.",
    )
    parser.add_argument(
        "--beta_warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for beta (0-1 of total steps).",
    )
    parser.add_argument(
        "--shrink_prob",
        type=float,
        default=0.1,
        help="Probability to zero out noisy_latents_inf (shrinkage).",
    )
    parser.add_argument(
        "--disable_augment",
        action="store_true",
        help="Disable random crop/flip/color jitter for alignment debugging.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_example_index",
        type=int,
        default=0,
        help="Zero-based validation JSONL row index used for the saved comparison image at every validation step.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--validation_jsonl",
        type=str,
        default=None,
        help="Path to a jsonl file containing image paths and captions for validation.",
    )
    parser.add_argument(
        "--multiple_datasets",
        type=str,
        nargs="+",
        help="A list of dataset names to be used for training.",
    )
    parser.add_argument(
        "--multiple_datasets_probabilities",
        type=float,
        nargs="+",
        help="A list of probabilities for each dataset to be used for training.",
    )
    parser.add_argument(
        "--resume_from_pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model to resume training from.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

class RandomRotate90:
    def __call__(self, image):
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return image
        return image.rotate(90 * k, expand=False)


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, probabilities, train_args):
        self.datasets = datasets
        self.probabilities = probabilities
        self.cumulative_probabilities = np.cumsum(probabilities)
        self.args = train_args

        self.resize_transform = transforms.Resize(
            self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        )
        if self.args.disable_augment:
            self.image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.prior_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.image_transforms = transforms.Compose([
                transforms.RandomCrop(self.args.resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotate90(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.prior_transforms = transforms.Compose([
                transforms.RandomCrop(self.args.resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotate90(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        rand = random.random()
        dataset_idx = np.searchsorted(self.cumulative_probabilities, rand)
        dataset = self.datasets[dataset_idx]
        item = dataset[idx % len(dataset)]

        image_path = item["image_path"]
        conditioning_image_path = item["conditioning_image_path"]
        prior_path = item["prior_path"]

        image = Image.open(image_path).convert("RGB")
        conditioning_image = Image.open(conditioning_image_path).convert("RGB")
        prior = np.load(prior_path)
        if prior.ndim > 2:
            prior = prior.squeeze()
        prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
        prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
        prior = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
        if self.args.use_m_local_diffusion:
            resolved_m_local_path = resolve_m_local_path(
                item,
                conditioning_image_path,
                self.args.m_local_dirs,
                self.args.m_local_column,
                self.args.m_local_missing_policy,
            )
            m_local = Image.new("L", conditioning_image.size, color=0) if resolved_m_local_path is None else load_map_pil(resolved_m_local_path)
        else:
            m_local = None

        image = self.resize_transform(image)
        conditioning_image = self.resize_transform(conditioning_image)
        prior = self.resize_transform(prior)
        if m_local is not None:
            m_local = self.resize_transform(m_local)
        if prior.size != conditioning_image.size:
            prior = prior.resize(conditioning_image.size, Image.Resampling.LANCZOS)
        if m_local is not None and m_local.size != conditioning_image.size:
            m_local = m_local.resize(conditioning_image.size, Image.Resampling.LANCZOS)

        seed = torch.random.seed()
        torch.manual_seed(seed)
        image = self.image_transforms(image)
        torch.manual_seed(seed)
        conditioning_image = self.image_transforms(conditioning_image)
        torch.manual_seed(seed)
        prior = self.prior_transforms(prior)
        if m_local is not None:
            torch.manual_seed(seed)
            m_local = self.prior_transforms(m_local)

        result = {
            "pixel_values": image,
            "conditioning_pixel_values": conditioning_image,
            "prior": prior,
            "input_ids": item["input_ids"],
        }
        if m_local is not None:
            result["m_local"] = m_local
        return result

def make_train_dataset(args, tokenizer, accelerator):
    datasets = []
    for dataset_name in args.multiple_datasets:
        dataset = load_dataset(
            args.train_data_dir,
            cache_dir=args.cache_dir,
            data_files={"train": f"{dataset_name}"},
        )
        datasets.append(dataset["train"])

    train_datasets = []
    for dataset in datasets:
        column_names = dataset.column_names
        image_column = args.image_column if args.image_column in column_names else column_names[0]
        caption_column = args.caption_column if args.caption_column in column_names else column_names[1]
        conditioning_image_column = args.conditioning_image_column if args.conditioning_image_column in column_names else column_names[2]
        prior_column = args.prior_column if args.prior_column in column_names else "prior"
        m_local_column = args.m_local_column if args.m_local_column in column_names else None

        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if random.random() < args.proportion_empty_prompts:
                    captions.append("")
                elif isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
            inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            return inputs.input_ids

        def preprocess_train(examples):
            examples["image_path"] = examples[image_column]
            examples["conditioning_image_path"] = examples[conditioning_image_column]
            examples["prior_path"] = examples[prior_column]
            if args.use_m_local_diffusion:
                examples["m_local_path"] = examples[m_local_column] if m_local_column is not None else [None] * len(examples[conditioning_image_column])
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
            train_dataset = dataset.with_transform(preprocess_train)
            train_datasets.append(train_dataset)

    return FuseDataset(train_datasets, args.multiple_datasets_probabilities, args)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prior = torch.stack([example["prior"] for example in examples])
    prior = prior.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    batch = {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prior": prior,
        "input_ids": input_ids,
    }
    if "m_local" in examples[0]:
        m_local = torch.stack([example["m_local"] for example in examples])
        batch["m_local"] = m_local.to(memory_format=torch.contiguous_format).float()
    return batch


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
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

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    if args.resume_from_pretrained:
        controlnet = ControlNetVAEModel.from_pretrained(args.resume_from_pretrained + "/controlnet")
        unet = UNet2DConditionModel.from_pretrained(args.resume_from_pretrained + "/unet")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )

        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = ControlNetVAEModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet")
            controlnet = ControlNetVAEModel.from_unet(unet)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unwrap_model(controlnet).save_pretrained(os.path.join(output_dir, 'controlnet'))
                unwrap_model(unet).save_pretrained(os.path.join(output_dir, 'unet'))

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, ControlNetVAEModel):
                    load_model = ControlNetVAEModel.from_pretrained(input_dir, subfolder="controlnet")
                elif isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                else:
                    raise ValueError(f"Unsupported model type in checkpoint load hook: {type(model)}")

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    for name, param in unet.named_parameters():
        if not "up_blocks" in name:
            param.requires_grad = False
    for name, param in controlnet.named_parameters():
        if "controlnet" in name:
            param.requires_grad = False
    text_encoder.requires_grad_(False)
    controlnet.train()
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(unet.parameters()) + list(controlnet.parameters())
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    ema_controlnet = None
    ema_unet = None
    if args.use_controlnet_ema:
        ema_controlnet = clone_ema_module(unwrap_model(controlnet)).to(accelerator.device)
        logger.info("Using ControlNet EMA with decay %.6f", args.ema_decay)
    if args.use_unet_ema:
        ema_unet = clone_ema_module(unwrap_model(unet)).to(accelerator.device)
        logger.info("Using UNet EMA with decay %.6f", args.ema_decay)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        setup_rank0_file_logging(args.output_dir)
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        tracker_config.pop("multiple_datasets")
        tracker_config.pop("multiple_datasets_probabilities")
        if tracker_config.get("m_local_dirs") is not None:
            tracker_config["m_local_dirs"] = ",".join(map(str, tracker_config["m_local_dirs"]))

        if args.report_to is not None:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"prior_scale = {args.prior_scale}")
    logger.info(f"beta_max = {args.beta_max}, beta_warmup_ratio = {args.beta_warmup_ratio}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            checkpoint_path = os.path.join(args.output_dir, path)
            accelerator.load_state(checkpoint_path)
            if args.use_controlnet_ema or args.use_unet_ema:
                loaded_controlnet_ema, loaded_unet_ema = load_diffusion_ema_checkpoint(
                    ema_controlnet,
                    ema_unet,
                    checkpoint_path,
                    torch_dtype=torch.float32,
                )
                if args.use_controlnet_ema and not loaded_controlnet_ema:
                    logger.info("No ControlNet EMA found in %s; initializing from resumed ControlNet weights", checkpoint_path)
                    copy_module_state(ema_controlnet, unwrap_model(controlnet))
                if args.use_unet_ema and not loaded_unet_ema:
                    logger.info("No UNet EMA found in %s; initializing from resumed UNet weights", checkpoint_path)
                    copy_module_state(ema_unet, unwrap_model(unet))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    best_validation_score = -float("inf")
    history = create_training_history()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                    dtype=weight_dtype
                )

                # add noise with full timestep
                timesteps_inf = torch.full_like(timesteps, noise_scheduler.config.num_train_timesteps - 1)
                noisy_latents_inf = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps_inf).to(
                    dtype=weight_dtype
                )

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Encode control image with VAE before passing to ControlNet.
                cond_latents = vae.encode(controlnet_image).latent_dist.sample()
                cond_latents = cond_latents * vae.config.scaling_factor

                prior_map = batch["prior"].to(dtype=weight_dtype).clamp(0.0, 1.0)
                if args.use_m_local_diffusion:
                    m_local_map = batch["m_local"].to(dtype=weight_dtype).clamp(0.0, 1.0)
                    prior_map = (prior_map + args.m_local_lambda * m_local_map).clamp(0.0, 1.0)

                hf_mag = compute_hf_mag(controlnet_image)

                # shrinkage switch
                if random.random() < args.shrink_prob:
                    noisy_latents_inf = torch.zeros_like(noisy_latents_inf)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    cond_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                beta = compute_beta(global_step, args.max_train_steps, args.beta_warmup_ratio, args.beta_max)
                if beta > 0.0:
                    prior_l = prior_map.to(device=cond_latents.device, dtype=cond_latents.dtype)
                    hf_l = hf_mag.to(device=cond_latents.device, dtype=cond_latents.dtype)
                    gated_down = []
                    for res in down_block_res_samples:
                        prior_res = F.interpolate(prior_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
                        hf_res = F.interpolate(hf_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
                        gate = 1.0 + beta * prior_res * hf_res
                        gate = gate.clamp(1.0, 1.0 + args.beta_max)
                        gated_down.append(res * gate)
                    down_block_res_samples = gated_down

                    prior_mid = F.interpolate(prior_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
                    hf_mid = F.interpolate(hf_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
                    gate_mid = 1.0 + beta * prior_mid * hf_mid
                    gate_mid = gate_mid.clamp(1.0, 1.0 + args.beta_max)
                    mid_block_res_sample = mid_block_res_sample * gate_mid

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents_inf,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noisy_latents.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    if args.use_controlnet_ema:
                        update_ema_module(ema_controlnet, unwrap_model(controlnet), args.ema_decay)
                    if args.use_unet_ema:
                        update_ema_module(ema_unet, unwrap_model(unet), args.ema_decay)
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if args.use_controlnet_ema or args.use_unet_ema:
                            save_diffusion_ema_checkpoint(ema_controlnet, ema_unet, save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_jsonl is not None and global_step % args.validation_steps == 0:
                        validation_unet = ema_unet if args.use_unet_ema else unet
                        validation_controlnet = ema_controlnet if args.use_controlnet_ema else controlnet
                        validation_output = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            validation_unet,
                            validation_controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        image_logs = validation_output["image_logs"]
                        metrics = validation_output["metrics"]
                        save_validation_examples(image_logs, args.output_dir, global_step)
                        if metrics is not None:
                            update_training_history(history, "validation", global_step, metrics)
                            save_training_history(history, args.output_dir)
                        if metrics is not None and metrics["final_score"] > best_validation_score:
                            best_validation_score = metrics["final_score"]
                            logger.info("New best final_score: %.4f. Saving best Stage 1 modules...", best_validation_score)
                            best_controlnet = ema_controlnet if args.use_controlnet_ema else unwrap_model(controlnet)
                            best_unet = ema_unet if args.use_unet_ema else unwrap_model(unet)
                            save_stage1_best_model(best_controlnet, best_unet, None, args.output_dir, global_step, metrics)

                if args.validation_jsonl is not None and global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process and global_step % args.log_interval == 0:
                update_training_history(history, "train", global_step, logs)
                save_training_history(history, args.output_dir)
                logger.info(
                    "Iteration [%d/%d] Loss: %.6f LR: %.8f",
                    global_step,
                    args.max_train_steps,
                    logs["loss"],
                    logs["lr"],
                )

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_controlnet = ema_controlnet if args.use_controlnet_ema else unwrap_model(controlnet)
        final_unet = ema_unet if args.use_unet_ema else unwrap_model(unet)
        save_stage1_modules(final_controlnet, final_unet, args.output_dir, accelerator=None)
        if args.use_controlnet_ema or args.use_unet_ema:
            raw_dir = Path(args.output_dir) / "raw_final"
            save_stage1_modules(unwrap_model(controlnet), unwrap_model(unet), raw_dir, accelerator=None)

        # Run a final round of validation.
        image_logs = None
        if args.validation_jsonl is not None:
            validation_output = log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )
            image_logs = validation_output["image_logs"]
            metrics = validation_output["metrics"]
            save_validation_examples(image_logs, args.output_dir, global_step)
            if metrics is not None:
                update_training_history(history, "validation", global_step, metrics)
                save_training_history(history, args.output_dir)
            if metrics is not None and metrics["final_score"] > best_validation_score:
                best_validation_score = metrics["final_score"]
                logger.info("New best final_score: %.4f. Saving best Stage 1 modules...", best_validation_score)
                best_controlnet = ema_controlnet if args.use_controlnet_ema else final_controlnet
                best_unet = ema_unet if args.use_unet_ema else final_unet
                save_stage1_best_model(best_controlnet, best_unet, None, args.output_dir, global_step, metrics)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

