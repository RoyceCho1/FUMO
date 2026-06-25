#!/usr/bin/env python
import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "experiment"


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def resolve_output_dir(config: dict, config_path: Path, write_marker: bool) -> Path:
    output_base = Path(config["paths"]["output_dir"])
    if output_base.name != "outputs":
        return output_base

    experiment_name = slugify(str(config.get("experiment", {}).get("name", "experiment")))
    env_run_id = os.environ.get("FUMO_RUN_ID")
    if env_run_id:
        return output_base / f"{experiment_name}_{slugify(env_run_id)}"

    local_rank = get_local_rank()
    master_port = os.environ.get("MASTER_PORT", "single")
    parent_pid = os.getppid()
    state_dir = output_base / ".run_state"
    marker_path = state_dir / f"{slugify(config_path.stem)}_{parent_pid}_{master_port}.txt"

    if not write_marker:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_base / f"{experiment_name}_{run_id}"

    state_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        marker_path.write_text(f"{experiment_name}_{run_id}", encoding="utf-8")
    else:
        for _ in range(300):
            if marker_path.exists():
                break
            time.sleep(0.1)
        if not marker_path.exists():
            raise TimeoutError(f"Timed out waiting for run directory marker: {marker_path}")

    run_name = marker_path.read_text(encoding="utf-8").strip()
    if not run_name:
        raise ValueError(f"Empty run directory marker: {marker_path}")
    return output_base / run_name


def add_arg(args, name, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(f"--{name}")
        return
    args.extend([f"--{name}", str(value)])


def main():
    parser = argparse.ArgumentParser(description="Launch FUMO Stage 1 training from a yaml config.")
    parser.add_argument("--config", default="config/baseline.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved train_diffusion args without running.")
    parsed = parser.parse_args()

    config_path = Path(parsed.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stage1 = config["stage1"]
    paths = config["paths"]
    json_dir = Path(paths["json_dir"])
    output_dir = resolve_output_dir(config, config_path, write_marker=not parsed.dry_run)

    train_args = []
    add_arg(train_args, "pretrained_model_name_or_path", stage1["pretrained_model_name_or_path"])
    add_arg(train_args, "train_data_dir", json_dir)
    train_args.extend(["--multiple_datasets", "baseline_train.jsonl"])
    train_args.extend(["--multiple_datasets_probabilities", "1.0"])
    add_arg(train_args, "validation_jsonl", json_dir / "baseline_val.jsonl")
    add_arg(train_args, "output_dir", output_dir)

    for key in (
        "resolution",
        "train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "max_train_steps",
        "checkpointing_steps",
        "log_interval",
        "checkpoints_total_limit",
        "validation_steps",
        "num_validation_images",
        "validation_example_index",
        "dataloader_num_workers",
        "mixed_precision",
        "report_to",
        "tracker_project_name",
        "lr_scheduler",
        "lr_warmup_steps",
        "beta_max",
        "beta_warmup_ratio",
        "shrink_prob",
        "seed",
        "enable_xformers_memory_efficient_attention",
        "gradient_checkpointing",
    ):
        add_arg(train_args, key, stage1.get(key))

    if parsed.dry_run:
        print("python train_diffusion.py " + " ".join(map(str, train_args)))
        return

    local_rank = get_local_rank()
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot = dict(config)
        config_snapshot["paths"] = dict(config["paths"])
        config_snapshot["paths"]["output_dir"] = str(output_dir)
        config_snapshot_path = output_dir / "config.yaml"
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_snapshot, f, sort_keys=False, allow_unicode=True)
        print(f"Resolved output_dir: {output_dir}")
        print(f"Saved config snapshot to {config_snapshot_path}")

    from train_diffusion import main as train_main, parse_args as parse_train_args

    train_main(parse_train_args([str(arg) for arg in train_args]))


if __name__ == "__main__":
    main()
