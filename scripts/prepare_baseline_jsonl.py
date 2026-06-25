#!/usr/bin/env python
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rows(lq_dir: Path, gt_dir: Path, prior_dir: Path, prompt: str):
    rows = []
    by_gt = defaultdict(list)
    missing = []
    for lq_path in sorted(p for p in lq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        gt_id = lq_path.stem.split("-")[0]
        gt_path = gt_dir / f"{gt_id}.png"
        prior_path = prior_dir / f"{lq_path.stem}.npy"
        if not gt_path.exists() or not prior_path.exists():
            missing.append({
                "lq": str(lq_path),
                "gt": str(gt_path),
                "prior": str(prior_path),
                "gt_exists": gt_path.exists(),
                "prior_exists": prior_path.exists(),
            })
            continue
        row = {
            "conditioning_image": str(lq_path.resolve()),
            "image": str(gt_path.resolve()),
            "prior": str(prior_path.resolve()),
            "text": prompt,
        }
        rows.append(row)
        by_gt[gt_id].append(row)
    return rows, by_gt, missing


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create GT-based train/val JSONL files for the FUMO baseline.")
    parser.add_argument("--config", default="config/baseline.yaml", help="Path to baseline yaml config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing json outputs.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    paths = config["paths"]
    split = config["split"]
    prompt = config.get("experiment", {}).get("prompt", "remove degradation")

    lq_dir = Path(paths["lq_dir"])
    gt_dir = Path(paths["gt_dir"])
    prior_dir = Path(paths["prior_dir"])
    json_dir = Path(paths["json_dir"])

    rows, by_gt, missing = build_rows(lq_dir, gt_dir, prior_dir, prompt)
    if missing:
        report_path = json_dir / "baseline_missing_paths.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(missing, indent=2), encoding="utf-8")
        raise FileNotFoundError(f"Found {len(missing)} rows with missing GT/prior paths. See {report_path}")

    gt_ids = sorted(by_gt)
    val_count = max(1, math.ceil(len(gt_ids) * float(split["val_ratio"])))
    rng = random.Random(int(split["seed"]))
    val_ids = sorted(rng.sample(gt_ids, val_count))
    train_ids = [gt_id for gt_id in gt_ids if gt_id not in set(val_ids)]

    train_rows = [row for gt_id in train_ids for row in by_gt[gt_id]]
    val_rows = [row for gt_id in val_ids for row in by_gt[gt_id]]

    outputs = {
        "train": json_dir / "baseline_train.jsonl",
        "val": json_dir / "baseline_val.jsonl",
        "split": json_dir / "baseline_split.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError("Output files already exist. Use --overwrite. Existing: " + ", ".join(existing))

    write_jsonl(outputs["train"], train_rows)
    write_jsonl(outputs["val"], val_rows)
    split_payload = {
        "seed": int(split["seed"]),
        "val_ratio": float(split["val_ratio"]),
        "num_gt": len(gt_ids),
        "num_val_gt": len(val_ids),
        "num_train_gt": len(train_ids),
        "num_total_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "train_gt_ids": train_ids,
        "val_gt_ids": val_ids,
    }
    outputs["split"].write_text(json.dumps(split_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(split_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
