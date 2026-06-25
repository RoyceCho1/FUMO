import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from wavelet_color_fix import wavelet_decomposition


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wavelet high-frequency prior P_hf from an image folder.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save P_hf .npy files.")
    parser.add_argument("--gray_output_dir", type=str, default=None, help="Folder to save grayscale P_hf PNGs.")
    parser.add_argument("--heatmap_output_dir", type=str, default=None, help="Folder to save color heatmap PNGs.")
    parser.add_argument("--overlay_output_dir", type=str, default=None, help="Folder to save input/P_hf overlay PNGs.")
    parser.add_argument("--levels", type=int, default=5, help="Wavelet decomposition levels.")
    parser.add_argument("--max_size", type=int, default=0, help="Resize long side before processing. 0 keeps original.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy and visualization files.")
    parser.add_argument("--no_gray", action="store_true", help="Do not save grayscale PNGs.")
    parser.add_argument("--no_heatmap", action="store_true", help="Do not save heatmap PNGs.")
    parser.add_argument("--no_overlay", action="store_true", help="Do not save overlay PNGs.")
    parser.add_argument("--overlay_alpha", type=float, default=0.35, help="Heatmap alpha for overlay PNGs.")
    return parser.parse_args()


def list_images(input_dir: str) -> list[str]:
    paths = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and Path(name).suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def resize_long_side(image: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return image
    scale = max_size / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_phf(image_bgr: np.ndarray, device: torch.device, levels: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        high_freq, _ = wavelet_decomposition(image, levels=levels)
        hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
        mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        phf = (hf_mag / mean).clamp(0.0, 1.0)
    return phf.squeeze().detach().cpu().numpy().astype(np.float32)


def prior_to_uint8(prior: np.ndarray) -> np.ndarray:
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    return (prior * 255.0 + 0.5).astype(np.uint8)


def save_visualizations(
    phf: np.ndarray,
    image_bgr: np.ndarray,
    stem: str,
    args: argparse.Namespace,
) -> None:
    gray = prior_to_uint8(phf)
    if gray.shape[:2] != image_bgr.shape[:2]:
        gray = cv2.resize(gray, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    if not args.no_gray and args.gray_output_dir:
        os.makedirs(args.gray_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.gray_output_dir, f"{stem}.png"), gray)

    color = None
    if (not args.no_heatmap and args.heatmap_output_dir) or (not args.no_overlay and args.overlay_output_dir):
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    if not args.no_heatmap and args.heatmap_output_dir:
        os.makedirs(args.heatmap_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.heatmap_output_dir, f"{stem}.png"), color)

    if not args.no_overlay and args.overlay_output_dir:
        os.makedirs(args.overlay_output_dir, exist_ok=True)
        overlay = cv2.addWeighted(image_bgr, 1.0 - args.overlay_alpha, color, args.overlay_alpha, 0)
        cv2.imwrite(os.path.join(args.overlay_output_dir, f"{stem}.png"), overlay)


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.gray_output_dir is None:
        args.gray_output_dir = f"{output_dir}_gray"
    if args.heatmap_output_dir is None:
        args.heatmap_output_dir = f"{output_dir}_heatmap"
    if args.overlay_output_dir is None:
        args.overlay_output_dir = f"{output_dir}_overlay"

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)

    processed = 0
    skipped = 0
    for image_path in tqdm(image_paths, desc="Generating P_hf"):
        stem = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{stem}.npy")
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[Skip] failed to read image: {image_path}")
            skipped += 1
            continue

        image_bgr = resize_long_side(image_bgr, args.max_size)
        phf = compute_phf(image_bgr, device, args.levels)
        np.save(output_path, phf)
        save_visualizations(phf, image_bgr, stem, args)
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, total={len(image_paths)}")


if __name__ == "__main__":
    main()
