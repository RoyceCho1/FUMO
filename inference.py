#!/usr/bin/env python
"""Full FUMO inference: LQ + P_int -> diffusion prelim -> refine final."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from basicsr.models.archs.NAFNet_arch import NAFNet
from fumo_refine_common import apply_refine_residual, infer_diff_prelim, load_pipeline, load_prior_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full FUMO inference with matched LQ/P_int files.")
    parser.add_argument("--baseline_config", default="config/baseline.yaml")
    parser.add_argument("--refine_config", default="config/refine_baseline.yaml")
    parser.add_argument("--input_dir", default=None, help="Directory containing LQ images.")
    parser.add_argument("--prior_dir", default=None, help="Directory containing P_int .npy files.")
    parser.add_argument("--output_dir", default="results/inference", help="Output directory under results.")
    parser.add_argument("--run_name", default=None, help="Optional subdirectory name inside output_dir.")
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--controlnet_dir", default=None)
    parser.add_argument("--unet_dir", default=None)
    parser.add_argument("--refine_dir", default=None, help="Run root or best/final directory for refine weights.")
    parser.add_argument("--refine_net_path", default=None)
    parser.add_argument("--refine_head_path", default=None)
    parser.add_argument("--use_final_refine", action="store_true", help="Use final refine weights instead of best.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--resolution_mode", default="square", choices=["square", "full"])
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--nafnet_width", type=int, default=None)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=None)
    parser.add_argument("--nafnet_enc_blk_nums", type=int, nargs="+", default=None)
    parser.add_argument("--nafnet_dec_blk_nums", type=int, nargs="+", default=None)
    parser.add_argument("--save_prelim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_input", action="store_true")
    parser.add_argument("--save_original_size", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()


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


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def shard_items(items: list[Path], num_shards: int, shard_id: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    if not 0 <= shard_id < num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < --num_shards.")
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_id]


def find_latest_refine_dir(outputs_dir: Path) -> Path | None:
    candidates = []
    for run_dir in outputs_dir.glob("refine_baseline_*"):
        best = run_dir / "best"
        if (best / "nafnet_refine.pth").exists() and (best / "nafnet_refine_head.pth").exists():
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_refine_paths(args: argparse.Namespace, refine_config: dict) -> tuple[Path, Path, Path | None]:
    if args.refine_net_path and args.refine_head_path:
        return Path(args.refine_net_path), Path(args.refine_head_path), None

    refine_dir = Path(args.refine_dir) if args.refine_dir else None
    if refine_dir is None:
        output_base = Path(refine_config.get("paths", {}).get("output_dir", "outputs"))
        if output_base.name != "outputs":
            output_base = output_base.parent
        refine_dir = find_latest_refine_dir(output_base)
        if refine_dir is None:
            raise FileNotFoundError(
                "Could not auto-detect refine weights. Pass --refine_dir or "
                "--refine_net_path/--refine_head_path."
            )

    if args.use_final_refine:
        net_path = refine_dir / "nafnet_refine_final.pth"
        head_path = refine_dir / "nafnet_refine_head_final.pth"
    else:
        if refine_dir.name == "best":
            best_dir = refine_dir
        else:
            best_dir = refine_dir / "best"
        net_path = best_dir / "nafnet_refine.pth"
        head_path = best_dir / "nafnet_refine_head.pth"

    if not net_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Missing refine weights: {net_path}, {head_path}")
    return net_path, head_path, refine_dir


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    return torch.float32


def load_refine_models(args: argparse.Namespace, net_path: Path, head_path: Path, device: torch.device):
    in_ch = 10
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)

    refine_net.load_state_dict(torch.load(net_path, map_location="cpu"))
    refine_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    refine_net.eval()
    refine_head.eval()
    return refine_net, refine_head


def image_to_tensor(
    path: Path,
    resolution: int,
    resolution_mode: str = "square",
    resize: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    original = Image.open(path).convert("RGB")
    original_size = original.size
    if resolution_mode == "square":
        target_size = (resolution, resolution)
    elif resize is not None:
        target_size = resize
    else:
        target_size = original_size
    image = original.resize(target_size, Image.Resampling.BILINEAR) if original.size != target_size else original
    return TF.to_tensor(image).unsqueeze(0), original_size, original


def tensor_to_image(tensor: torch.Tensor, original_size: tuple[int, int] | None = None) -> Image.Image:
    image = TF.to_pil_image(tensor.detach().float().cpu().clamp(0.0, 1.0).squeeze(0))
    if original_size is not None and image.size != original_size:
        image = image.resize(original_size, Image.Resampling.BILINEAR)
    return image


def save_metadata(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "inference_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    baseline_config = load_yaml(args.baseline_config)
    refine_config = load_yaml(args.refine_config)
    baseline_paths = baseline_config.get("paths", {})
    refine_paths = refine_config.get("paths", {})
    stage1 = baseline_config.get("stage1", {})
    stage2 = refine_config.get("stage2", {})

    args.input_dir = first_existing(args.input_dir, baseline_paths.get("lq_dir"))
    args.prior_dir = first_existing(args.prior_dir, baseline_paths.get("prior_dir"))
    args.pretrained_model_name_or_path = first_existing(
        args.pretrained_model_name_or_path,
        refine_paths.get("pretrained_model_name_or_path"),
        stage1.get("pretrained_model_name_or_path"),
    )
    args.controlnet_dir = first_existing(args.controlnet_dir, refine_paths.get("controlnet_dir"))
    args.unet_dir = first_existing(args.unet_dir, refine_paths.get("unet_dir"))
    args.prompt = first_existing(
        args.prompt,
        refine_config.get("experiment", {}).get("prompt"),
        baseline_config.get("experiment", {}).get("prompt"),
        "remove degradation",
    )
    validation_cfg = refine_config.get("validation", {})
    args.resolution = int(first_existing(args.resolution, stage2.get("resolution"), stage1.get("resolution"), 768))
    args.resolution_mode = first_existing(args.resolution_mode, validation_cfg.get("resolution_mode"), "square")
    args.resize = first_existing(args.resize, validation_cfg.get("resize"))
    if args.resize is not None:
        args.resize = tuple(int(v) for v in args.resize)
    args.beta = float(first_existing(args.beta, stage2.get("beta"), stage1.get("beta_max"), 0.25))
    args.nafnet_width = int(first_existing(args.nafnet_width, stage2.get("nafnet_width"), 64))
    args.nafnet_middle_blk_num = int(first_existing(args.nafnet_middle_blk_num, stage2.get("nafnet_middle_blk_num"), 1))
    args.nafnet_enc_blk_nums = first_existing(args.nafnet_enc_blk_nums, stage2.get("nafnet_enc_blk_nums"), [1, 1, 1, 28])
    args.nafnet_dec_blk_nums = first_existing(args.nafnet_dec_blk_nums, stage2.get("nafnet_dec_blk_nums"), [1, 1, 1, 1])

    required = {
        "input_dir": args.input_dir,
        "prior_dir": args.prior_dir,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "controlnet_dir": args.controlnet_dir,
        "unet_dir": args.unet_dir,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required paths: {', '.join(missing)}")

    input_dir = Path(args.input_dir)
    prior_dir = Path(args.prior_dir)
    run_root = Path(args.output_dir)
    if args.run_name:
        run_root = run_root / args.run_name
    prelim_dir = run_root / "prelim"
    final_dir = run_root / "final"
    input_save_dir = run_root / "input"
    for directory, enabled in ((prelim_dir, args.save_prelim), (final_dir, args.save_final), (input_save_dir, args.save_input)):
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    images = shard_items(images, args.num_shards, args.shard_id)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    net_path, head_path, resolved_refine_dir = resolve_refine_paths(args, refine_config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)

    pipeline_args = SimpleNamespace(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        controlnet_dir=args.controlnet_dir,
        unet_dir=args.unet_dir,
    )

    print(f"Input dir: {input_dir}")
    print(f"Prior dir: {prior_dir}")
    print(f"Output dir: {run_root}")
    print(f"Base model: {args.pretrained_model_name_or_path}")
    print(f"ControlNet: {args.controlnet_dir}")
    print(f"UNet: {args.unet_dir}")
    print(f"Refine net: {net_path}")
    print(f"Refine head: {head_path}")
    print(f"Images: {len(images)} shard {args.shard_id}/{args.num_shards}")
    print(f"Resolution mode: {args.resolution_mode} resize={args.resize} processing_resolution={args.resolution}")

    save_metadata(
        run_root,
        {
            "input_dir": str(input_dir),
            "prior_dir": str(prior_dir),
            "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
            "controlnet_dir": str(args.controlnet_dir),
            "unet_dir": str(args.unet_dir),
            "refine_dir": str(resolved_refine_dir) if resolved_refine_dir else None,
            "refine_net_path": str(net_path),
            "refine_head_path": str(head_path),
            "prompt": args.prompt,
            "resolution": args.resolution,
            "resolution_mode": args.resolution_mode,
            "resize": args.resize,
            "beta": args.beta,
            "residual_scale": args.residual_scale,
            "dtype": args.dtype,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
        },
    )

    pipeline = load_pipeline(pipeline_args, device, dtype)
    pipeline.default_processing_resolution = args.resolution
    refine_net, refine_head = load_refine_models(args, net_path, head_path, device)

    missing_priors = []
    processed_count = 0
    skipped_existing_count = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for image_path in tqdm(images, desc="FUMO inference"):
            relative = image_path.relative_to(input_dir)
            stem = relative.with_suffix("")
            prior_path = prior_dir / f"{stem.as_posix()}.npy"
            if not prior_path.exists():
                prior_path = prior_dir / f"{image_path.stem}.npy"
            if not prior_path.exists():
                missing_priors.append(str(image_path))
                continue

            final_path = final_dir / relative.with_suffix(".png")
            prelim_path = prelim_dir / relative.with_suffix(".png")
            if args.skip_existing and args.save_final and final_path.exists():
                skipped_existing_count += 1
                continue

            cond, original_size, original_image = image_to_tensor(
                image_path,
                args.resolution,
                resolution_mode=args.resolution_mode,
                resize=args.resize,
            )
            prior = load_prior_tensor(str(prior_path)).unsqueeze(0)
            prior = torch.nn.functional.interpolate(
                prior,
                size=cond.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cond = cond.to(device=device)
            prior = prior.to(device=device)

            prelim_pred = infer_diff_prelim(pipeline, cond, prior, args.prompt, args.beta)
            prelim = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0)
            prelim = prelim.float()
            cond = cond.float()
            prior = prior.float()
            refined = apply_refine_residual(
                refine_net,
                refine_head,
                prelim,
                cond,
                prior,
                residual_scale=args.residual_scale,
            )

            target_size = original_size if args.save_original_size else None
            if args.save_prelim:
                prelim_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(prelim, target_size).save(prelim_path)
            if args.save_final:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_image(refined, target_size).save(final_path)
            if args.save_input:
                input_path = input_save_dir / relative.with_suffix(".png")
                input_path.parent.mkdir(parents=True, exist_ok=True)
                if args.save_original_size:
                    original_image.save(input_path)
                else:
                    tensor_to_image(cond).save(input_path)
            processed_count += 1

    elapsed = time.perf_counter() - start_time
    seconds_per_image = elapsed / processed_count if processed_count > 0 else 0.0
    print(
        f"Inference done: processed={processed_count}, "
        f"missing_priors={len(missing_priors)}, skipped_existing={skipped_existing_count}, "
        f"elapsed={elapsed:.2f}s, sec/image={seconds_per_image:.3f}"
    )

    if missing_priors:
        missing_path = run_root / "missing_priors.txt"
        missing_path.write_text("\n".join(missing_priors) + "\n", encoding="utf-8")
        print(f"Warning: skipped {len(missing_priors)} images with missing P_int. See {missing_path}")


if __name__ == "__main__":
    main()
