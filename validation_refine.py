#!/usr/bin/env python
"""Standalone FUMO refine validation.

This script loads a diffusion checkpoint, refine checkpoint and validation JSONL,
runs full-image sampling, computes PSNR/SSIM/LPIPS/final_score, and saves grids.
"""

import argparse
import json
from pathlib import Path

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fumo_refine_common import (
    ValidationJsonlDataset,
    apply_refine_residual,
    calculate_validation_metrics,
    dtype_from_arg,
    first_existing,
    labeled_image_grid,
    load_jsonl,
    load_pipeline,
    load_refine_models,
    load_yaml,
    make_pipeline_args,
    resolve_refine_paths,
    tensor_to_pil,
    infer_diff_prelim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FUMO refine output on a JSONL split.")
    parser.add_argument("--baseline_config", default="config/baseline.yaml")
    parser.add_argument("--refine_config", default="config/refine.yaml")
    parser.add_argument("--validation_jsonl", default=None)
    parser.add_argument("--output_dir", default="results/validation_refine")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--controlnet_dir", default=None)
    parser.add_argument("--unet_dir", default=None)
    parser.add_argument("--refine_dir", default=None)
    parser.add_argument("--refine_net_path", default=None)
    parser.add_argument("--refine_head_path", default=None)
    parser.add_argument("--use_final_refine", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--resolution_mode", default=None, choices=["square", "full"])
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--example_index", type=int, default=0)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--use_m_local_diffusion", action="store_true", default=None)
    parser.add_argument("--use_m_local_refine", action="store_true", default=None)
    parser.add_argument("--m_local_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--m_local_column", default=None)
    parser.add_argument("--m_local_lambda", type=float, default=None)
    parser.add_argument("--m_local_missing_policy", default=None, choices=["error", "zero"])
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--nafnet_width", type=int, default=None)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=None)
    parser.add_argument("--nafnet_enc_blk_nums", type=int, nargs="+", default=None)
    parser.add_argument("--nafnet_dec_blk_nums", type=int, nargs="+", default=None)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    baseline_config = load_yaml(args.baseline_config)
    refine_config = load_yaml(args.refine_config)
    baseline_paths = baseline_config.get("paths", {})
    refine_paths = refine_config.get("paths", {})
    stage1 = baseline_config.get("stage1", {})
    stage2 = refine_config.get("stage2", {})
    validation = refine_config.get("validation", {})

    json_dir = Path(refine_paths.get("json_dir", "json"))
    args.validation_jsonl = first_existing(args.validation_jsonl, refine_paths.get("validation_jsonl"), json_dir / "baseline_val.jsonl")
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
    args.resolution = int(first_existing(args.resolution, stage2.get("resolution"), stage1.get("resolution"), 768))
    args.resolution_mode = first_existing(args.resolution_mode, validation.get("resolution_mode"), "square")
    args.resize = first_existing(args.resize, validation.get("resize"))
    args.num_images = first_existing(args.num_images, validation.get("num_images"))
    args.batch_size = int(first_existing(args.batch_size, validation.get("batch_size"), stage2.get("validation_batch_size"), 1))
    args.num_workers = int(first_existing(args.num_workers, validation.get("num_workers"), stage2.get("num_workers"), 4))
    args.beta = float(first_existing(args.beta, stage2.get("beta"), stage1.get("beta_max"), 0.25))
    args.use_m_local_diffusion = bool(first_existing(args.use_m_local_diffusion, stage2.get("use_m_local_diffusion"), stage1.get("use_m_local_diffusion"), False))
    args.use_m_local_refine = bool(first_existing(args.use_m_local_refine, stage2.get("use_m_local_refine"), False))
    args.m_local_dirs = first_existing(args.m_local_dirs, refine_paths.get("m_local_dirs"), baseline_paths.get("m_local_dirs"), [])
    args.m_local_column = first_existing(args.m_local_column, stage2.get("m_local_column"), stage1.get("m_local_column"), "m_local")
    args.m_local_lambda = float(first_existing(args.m_local_lambda, stage2.get("m_local_lambda"), stage1.get("m_local_lambda"), 0.5))
    args.m_local_missing_policy = first_existing(args.m_local_missing_policy, stage2.get("m_local_missing_policy"), stage1.get("m_local_missing_policy"), "error")
    args.nafnet_width = int(first_existing(args.nafnet_width, stage2.get("nafnet_width"), 64))
    args.nafnet_middle_blk_num = int(first_existing(args.nafnet_middle_blk_num, stage2.get("nafnet_middle_blk_num"), 1))
    args.nafnet_enc_blk_nums = first_existing(args.nafnet_enc_blk_nums, stage2.get("nafnet_enc_blk_nums"), [1, 1, 1, 28])
    args.nafnet_dec_blk_nums = first_existing(args.nafnet_dec_blk_nums, stage2.get("nafnet_dec_blk_nums"), [1, 1, 1, 1])
    args._refine_config = refine_config

    missing = [
        name for name in ("validation_jsonl", "pretrained_model_name_or_path", "controlnet_dir", "unet_dir")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    return args


def save_validation_grid(image_log: dict, output_dir: Path) -> None:
    if not image_log:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = labeled_image_grid(
        [image_log["cond"], image_log["prelim"], image_log["prediction"], image_log["gt"]],
        ["LQ", "prelim", "prediction", "GT"],
    )
    grid.save(output_dir / "validation_grid.png")


def main() -> None:
    args = resolve_args(parse_args())
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)
    output_dir = Path(args.output_dir)
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    net_path, head_path, resolved_refine_dir = resolve_refine_paths(
        args.refine_dir,
        args.refine_net_path,
        args.refine_head_path,
        args._refine_config,
        use_final_refine=args.use_final_refine,
    )

    print(f"Validation JSONL: {args.validation_jsonl}")
    print(f"Output dir: {output_dir}")
    print(f"Base model: {args.pretrained_model_name_or_path}")
    print(f"ControlNet: {args.controlnet_dir}")
    print(f"UNet: {args.unet_dir}")
    print(f"Refine net: {net_path}")
    print(f"Refine head: {head_path}")
    print(f"Validation mode: {args.resolution_mode} resize={args.resize} num_images={args.num_images}")

    entries = load_jsonl(args.validation_jsonl)
    dataset = ValidationJsonlDataset(
        entries,
        resolution=args.resolution,
        resolution_mode=args.resolution_mode,
        resize=args.resize,
        num_images=args.num_images,
        load_m_local=args.use_m_local_diffusion or args.use_m_local_refine,
        m_local_dirs=args.m_local_dirs,
        m_local_column=args.m_local_column,
        m_local_missing_policy=args.m_local_missing_policy,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pipeline_args = make_pipeline_args(args.pretrained_model_name_or_path, args.controlnet_dir, args.unet_dir)
    pipeline = load_pipeline(pipeline_args, device, dtype)
    pipeline.default_processing_resolution = args.resolution
    refine_net, refine_head = load_refine_models(args, net_path, head_path, device)

    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in lpips_metric.parameters():
        parameter.requires_grad_(False)

    metric_sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "final_score": 0.0}
    count = 0
    image_log = None
    seen = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="FUMO refine validation"):
            cond = batch["cond"].to(device)
            gt = batch["gt"].to(device)
            prior = batch["prior"].to(device)
            m_local = batch.get("m_local")
            if m_local is not None:
                m_local = m_local.to(device)
            prelim_pred = infer_diff_prelim(
                pipeline,
                cond,
                prior,
                args.prompt,
                args.beta,
                m_local_tensor=m_local if args.use_m_local_diffusion else None,
                m_local_lambda=args.m_local_lambda,
            )
            prelim = ((prelim_pred + 1.0) / 2.0).clamp(0.0, 1.0).float()
            cond = cond.float()
            prior = prior.float()
            pred = apply_refine_residual(
                refine_net,
                refine_head,
                prelim,
                cond,
                prior,
                m_local=m_local.float() if (args.use_m_local_refine and m_local is not None) else None,
                residual_scale=args.residual_scale,
            )

            metrics = calculate_validation_metrics(pred, gt, lpips_metric)
            batch_size = cond.shape[0]
            for key, value in metrics.items():
                metric_sums[key] += value * batch_size
            count += batch_size

            if image_log is None and seen <= args.example_index < seen + batch_size:
                local_idx = args.example_index - seen
                image_log = {
                    "cond": tensor_to_pil(cond[local_idx]),
                    "prelim": tensor_to_pil(prelim[local_idx]),
                    "prediction": tensor_to_pil(pred[local_idx]),
                    "gt": tensor_to_pil(gt[local_idx]),
                }
            seen += batch_size

    if count == 0:
        raise ValueError("No validation images were processed.")

    scores = {key: value / count for key, value in metric_sums.items()}
    scores.update({
        "num_images": count,
        "resolution_mode": args.resolution_mode,
        "resize": args.resize,
        "refine_dir": str(resolved_refine_dir) if resolved_refine_dir else None,
        "refine_net_path": str(net_path),
        "refine_head_path": str(head_path),
    })
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    save_validation_grid(image_log, output_dir)

    print(
        "PSNR: {psnr:.4f} SSIM: {ssim:.4f} LPIPS: {lpips:.4f} Final: {final_score:.4f}".format(**scores)
    )
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
