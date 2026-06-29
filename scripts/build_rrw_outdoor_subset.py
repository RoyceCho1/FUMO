#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTDOOR_KEYS = ("outdoor", "wild_out", "park", "trees", "plants", "car")


def parse_args():
    parser = argparse.ArgumentParser(description="Build RRW outdoor subset for FUMO training.")
    parser.add_argument("--rrw_root", default="/home/student_1/LoViF/LOVIF_repo/data/RRW")
    parser.add_argument("--json_dir", default="/home/student_1/LoViF/FUMO/json/rrw_outdoor_50_qwen3")
    parser.add_argument("--flat_lq_dir", default="/home/student_1/LoViF/FUMO/data/RRW_outdoor_50/LQ")
    parser.add_argument("--prior_dir", default="/home/student_1/LoViF/FUMO/results/RRW_outdoor_50/Qwen3-VL-8B/P_int")
    parser.add_argument("--frames_per_scene", type=int, default=50)
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--overwrite_links", action="store_true")
    return parser.parse_args()


def list_images(directory: Path):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_gt_dir(group_dir: Path):
    gt_dirs = [p for p in group_dir.iterdir() if p.is_dir() and p.name.lower().startswith("gt")]
    if not gt_dirs:
        return None
    if len(gt_dirs) > 1:
        gt_dirs = sorted(gt_dirs)
    return gt_dirs[0]


def is_outdoor_scene(name: str) -> bool:
    lname = name.lower()
    return any(key in lname for key in OUTDOOR_KEYS)


def sample_evenly(paths, count: int):
    if len(paths) <= count:
        return paths
    if count <= 1:
        return [paths[0]]
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    # round can duplicate on tiny lists; preserve order and top up if needed.
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
    if len(unique) < count:
        for idx in range(len(paths)):
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
            if len(unique) == count:
                break
    return [paths[idx] for idx in unique[:count]]


def main():
    args = parse_args()
    rrw_root = Path(args.rrw_root)
    json_dir = Path(args.json_dir)
    flat_lq_dir = Path(args.flat_lq_dir)
    prior_dir = Path(args.prior_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    flat_lq_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "rrw_root": str(rrw_root),
        "frames_per_scene": args.frames_per_scene,
        "flat_lq_dir": str(flat_lq_dir),
        "prior_dir": str(prior_dir),
        "scenes": [],
        "skipped_scenes": [],
    }

    for group_dir in sorted(p for p in rrw_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        gt_dir = find_gt_dir(group_dir)
        if gt_dir is None:
            continue
        for scene_dir in sorted(p for p in group_dir.iterdir() if p.is_dir() and not p.name.lower().startswith("gt")):
            if not is_outdoor_scene(scene_dir.name):
                continue
            gt_path = gt_dir / f"{scene_dir.name}_GT.png"
            if not gt_path.exists():
                summary["skipped_scenes"].append({
                    "group": group_dir.name,
                    "scene": scene_dir.name,
                    "reason": f"missing GT: {gt_path}",
                })
                continue
            images = list_images(scene_dir)
            if not images:
                summary["skipped_scenes"].append({"group": group_dir.name, "scene": scene_dir.name, "reason": "no images"})
                continue
            sampled = sample_evenly(images, args.frames_per_scene)
            scene_rows = 0
            for image_path in sampled:
                flat_stem = f"{group_dir.name}__{scene_dir.name}__{image_path.stem}"
                flat_path = flat_lq_dir / f"{flat_stem}{image_path.suffix.lower()}"
                if flat_path.exists() or flat_path.is_symlink():
                    if args.overwrite_links:
                        flat_path.unlink()
                if not flat_path.exists():
                    os.symlink(image_path, flat_path)
                rows.append({
                    "conditioning_image": str(flat_path),
                    "image": str(gt_path),
                    "prior": str(prior_dir / f"{flat_stem}.npy"),
                    "text": args.prompt,
                    "source": "RRW",
                    "rrw_group": group_dir.name,
                    "rrw_scene": scene_dir.name,
                    "rrw_frame": image_path.name,
                })
                scene_rows += 1
            summary["scenes"].append({
                "group": group_dir.name,
                "scene": scene_dir.name,
                "gt": str(gt_path),
                "available_frames": len(images),
                "sampled_frames": scene_rows,
            })

    jsonl_path = json_dir / "rrw_outdoor_50_train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary.update({
        "num_scenes": len(summary["scenes"]),
        "num_rows": len(rows),
        "jsonl": str(jsonl_path),
    })
    with (json_dir / "rrw_outdoor_50_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Wrote {jsonl_path}")
    print(f"Rows: {len(rows)}")
    print(f"Scenes: {len(summary['scenes'])}")
    print(f"Skipped scenes: {len(summary['skipped_scenes'])}")
    print(f"Flat LQ dir: {flat_lq_dir}")
    print(f"Expected P_int dir: {prior_dir}")


if __name__ == "__main__":
    main()
