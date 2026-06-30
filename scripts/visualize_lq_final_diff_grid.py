#!/usr/bin/env python
"""Create LQ vs final-output difference grids for visual inspection."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LQ - final differences as labeled image grids.")
    parser.add_argument(
        "--lq_dir",
        default="/home/student_1/LoViF/LOVIF_repo/data/RDRF_dataset/validation/LQ_RaindropRemoval_WeatherRemoval",
    )
    parser.add_argument(
        "--final_dir",
        default="/home/student_1/LoViF/FUMO/results/LQ_RaindropRemoval_WeatherRemoval_val/FUMO_tta_d4/final",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/student_1/LoViF/FUMO/results/LQ_RaindropRemoval_WeatherRemoval_val/FUMO_tta_d4/diff_grid",
    )
    parser.add_argument("--recursive", action="store_true", help="Match files recursively by relative path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Mask threshold in [0, 1].")
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Auto threshold percentile used when --threshold is not set.",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--diff_scale", type=float, default=1.0, help="Multiplier for diff heatmap visualization.")
    parser.add_argument("--save_diff_npy", action="store_true")
    return parser.parse_args()


def list_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def heatmap_from_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    vis = np.clip(gray * scale, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(to_uint8(vis), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def make_mask(diff_gray: np.ndarray, threshold: float | None, percentile: float) -> tuple[np.ndarray, float]:
    if threshold is None:
        threshold = float(np.percentile(diff_gray, percentile))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    mask = (diff_gray >= threshold).astype(np.float32)
    return mask, threshold


def make_overlay(lq: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    red = np.zeros_like(lq)
    red[..., 0] = 1.0
    mask3 = mask[..., None]
    return lq * (1.0 - alpha * mask3) + red * (alpha * mask3)


def draw_label(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, image.width // 42))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad_x = max(8, image.width // 150)
    pad_y = max(5, image.height // 150)
    bg = (0, 0, 0)
    draw.rectangle(
        (0, 0, bbox[2] - bbox[0] + 2 * pad_x, bbox[3] - bbox[1] + 2 * pad_y),
        fill=bg,
    )
    draw.text((pad_x, pad_y), label, fill=(255, 255, 255), font=font)
    return image


def make_grid(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    pil_images = [draw_label(Image.fromarray(to_uint8(image)), label) for image, label in zip(images, labels)]
    widths = [image.width for image in pil_images]
    heights = [image.height for image in pil_images]
    grid = Image.new("RGB", (sum(widths), max(heights)), color=(0, 0, 0))
    x = 0
    for image in pil_images:
        grid.paste(image, (x, 0))
        x += image.width
    return grid


def resolve_final_path(final_dir: Path, lq_dir: Path, lq_path: Path, recursive: bool) -> Path | None:
    if recursive:
        candidate = final_dir / lq_path.relative_to(lq_dir).with_suffix(".png")
        if candidate.exists():
            return candidate
    for ext in IMAGE_EXTS:
        candidate = final_dir / f"{lq_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    lq_dir = Path(args.lq_dir)
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lq_paths = list_images(lq_dir, args.recursive)
    if args.limit is not None:
        lq_paths = lq_paths[: args.limit]

    processed = 0
    missing = []
    for lq_path in tqdm(lq_paths, desc="Diff grids"):
        final_path = resolve_final_path(final_dir, lq_dir, lq_path, args.recursive)
        if final_path is None:
            missing.append(str(lq_path))
            continue

        lq = load_rgb(lq_path)
        final = load_rgb(final_path)
        if final.shape[:2] != lq.shape[:2]:
            final_u8 = cv2.resize(to_uint8(final), (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_LINEAR)
            final = final_u8.astype(np.float32) / 255.0

        diff = np.abs(lq - final)
        diff_gray = diff.max(axis=2)
        heatmap = heatmap_from_gray(diff_gray, args.diff_scale).astype(np.float32) / 255.0
        mask, threshold = make_mask(diff_gray, args.threshold, args.percentile)
        mask_rgb = np.repeat(mask[..., None], 3, axis=2)
        overlay = make_overlay(lq, mask, args.overlay_alpha)

        grid = make_grid(
            [lq, final, heatmap, mask_rgb, overlay],
            ["LQ", "final", "abs diff", f"mask >= {threshold:.3f}", "overlay"],
        )

        if args.recursive:
            out_path = output_dir / lq_path.relative_to(lq_dir).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = output_dir / f"{lq_path.stem}.png"
        grid.save(out_path)

        if args.save_diff_npy:
            np.save(out_path.with_suffix(".npy"), diff_gray.astype(np.float32))
        processed += 1

    if missing:
        missing_path = output_dir / "missing_final.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing final images: {len(missing)}. See {missing_path}")
    print(f"Saved diff grids: {processed} -> {output_dir}")


if __name__ == "__main__":
    main()
