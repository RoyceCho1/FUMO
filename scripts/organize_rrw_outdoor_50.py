#!/usr/bin/env python
import json
import os
from pathlib import Path


FUMO_ROOT = Path("/home/student_1/LoViF/FUMO")
RRW_ROOT = Path("/home/student_1/LoViF/LOVIF_repo/data/RRW")
OLD_LQ_DIR = FUMO_ROOT / "data" / "RRW_outdoor_50" / "LQ"
NEW_ROOT = RRW_ROOT / "outdoor_50"
NEW_LQ_DIR = NEW_ROOT / "LQ"
NEW_GT_DIR = NEW_ROOT / "GT"

JSONL_FILES = [
    FUMO_ROOT / "json" / "lovif_qwen25_rrw_outdoor_50" / "rrw_outdoor_50_train.jsonl",
    FUMO_ROOT / "json" / "lovif_qwen25_rrw_outdoor_50" / "train.jsonl",
    FUMO_ROOT / "json" / "lovif_qwen3_rrw_outdoor_50" / "rrw_outdoor_50_train.jsonl",
    FUMO_ROOT / "json" / "rrw_outdoor_50_qwen3" / "rrw_outdoor_50_train.jsonl",
]

SUMMARY_FILES = [
    FUMO_ROOT / "json" / "lovif_qwen25_rrw_outdoor_50" / "summary.json",
    FUMO_ROOT / "json" / "lovif_qwen3_rrw_outdoor_50" / "rrw_outdoor_50_summary.json",
    FUMO_ROOT / "json" / "rrw_outdoor_50_qwen3" / "rrw_outdoor_50_summary.json",
]


def is_rrw_row(row: dict) -> bool:
    cond = str(row.get("conditioning_image", ""))
    return row.get("source") == "RRW" or "/RRW_outdoor_50/" in cond or "/data/RRW/" in cond


def force_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(src, dst)


def rrw_gt_name(row: dict) -> str:
    group = row.get("rrw_group")
    scene = row.get("rrw_scene")
    gt_path = Path(row["image"])
    if group and scene:
        return f"{group}__{scene}_GT{gt_path.suffix}"
    return gt_path.name


def link_row_files(row: dict) -> dict:
    old_lq = Path(row["conditioning_image"])
    old_gt = Path(row["image"])

    src_lq = old_lq.resolve()
    src_gt = old_gt.resolve()
    new_lq = NEW_LQ_DIR / old_lq.name
    new_gt = NEW_GT_DIR / rrw_gt_name(row)

    force_symlink(src_lq, new_lq)
    force_symlink(src_gt, new_gt)

    row["conditioning_image"] = str(new_lq)
    row["image"] = str(new_gt)
    row["rrw_subset"] = "outdoor_50"
    return row


def rewrite_jsonl(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    rows = []
    total = 0
    rrw = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if is_rrw_row(row):
                row = link_row_files(row)
                rrw += 1
            rows.append(row)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total, rrw


def rewrite_summary(path: Path) -> None:
    if not path.exists():
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    data["rrw_root"] = str(RRW_ROOT)
    data["outdoor_50_root"] = str(NEW_ROOT)
    data["flat_lq_dir"] = str(NEW_LQ_DIR)
    data["gt_dir"] = str(NEW_GT_DIR)

    for scene in data.get("scenes", []):
        group = scene.get("group")
        scene_name = scene.get("scene")
        gt = scene.get("gt")
        if group and scene_name and gt:
            old_gt = Path(gt)
            scene["original_gt"] = str(old_gt)
            scene["gt"] = str(NEW_GT_DIR / f"{group}__{scene_name}_GT{old_gt.suffix}")

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate() -> dict:
    lq_count = len(list(NEW_LQ_DIR.glob("*")))
    gt_count = len(list(NEW_GT_DIR.glob("*")))
    missing = []

    for jsonl in JSONL_FILES:
        if not jsonl.exists():
            continue
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not is_rrw_row(row):
                    continue
                for key in ("conditioning_image", "image", "prior"):
                    value = row.get(key)
                    if value and not Path(value).exists():
                        missing.append({"jsonl": str(jsonl), "key": key, "path": value})
                        if len(missing) >= 20:
                            break
                if len(missing) >= 20:
                    break

    return {
        "new_lq_dir": str(NEW_LQ_DIR),
        "new_gt_dir": str(NEW_GT_DIR),
        "new_lq_count": lq_count,
        "new_gt_count": gt_count,
        "missing_examples": missing,
    }


def main() -> None:
    NEW_LQ_DIR.mkdir(parents=True, exist_ok=True)
    NEW_GT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for jsonl in JSONL_FILES:
        total, rrw = rewrite_jsonl(jsonl)
        results[str(jsonl)] = {"total_rows": total, "rrw_rows_rewritten": rrw}

    for summary in SUMMARY_FILES:
        rewrite_summary(summary)

    report = {
        "jsonl_updates": results,
        "validation": validate(),
        "old_lq_dir_left_in_place": str(OLD_LQ_DIR),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
