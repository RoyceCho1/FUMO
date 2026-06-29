#!/usr/bin/env python
import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "experiment"


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


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


def add_arg(args: list, name: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            args.append(f"--{name}")
        return
    if isinstance(value, (list, tuple)):
        args.append(f"--{name}")
        args.extend(str(item) for item in value)
        return
    args.extend([f"--{name}", str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FUMO refine training from a YAML config.")
    parser.add_argument("--config", default="config/refine_baseline.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved train_refine args without running.")
    parsed = parser.parse_args()

    config_path = Path(parsed.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    config = load_config(config_path)
    paths = config["paths"]
    stage2 = config["stage2"]
    validation = config.get("validation", {})
    output_dir = resolve_output_dir(config, config_path, write_marker=not parsed.dry_run)
    json_dir = Path(paths["json_dir"])
    train_jsonls = paths.get("train_jsonls", paths.get("train_jsonl", "baseline_train.jsonl"))
    if isinstance(train_jsonls, str):
        train_jsonls = [train_jsonls]
    validation_enabled = bool(validation.get("enabled", True))
    validation_jsonl = paths.get("validation_jsonl")
    if validation_jsonl is None:
        validation_jsonl = json_dir / "baseline_val.jsonl"

    train_args = []
    add_arg(train_args, "train_data_dir", json_dir)
    add_arg(train_args, "multiple_datasets", train_jsonls)
    add_arg(train_args, "multiple_datasets_probabilities", stage2.get("multiple_datasets_probabilities", [1.0 / len(train_jsonls)] * len(train_jsonls)))
    if validation_enabled:
        add_arg(train_args, "validation_jsonl", validation_jsonl)
    add_arg(train_args, "output_dir", output_dir)
    add_arg(train_args, "pretrained_model_name_or_path", paths["pretrained_model_name_or_path"])
    add_arg(train_args, "controlnet_dir", paths["controlnet_dir"])
    add_arg(train_args, "unet_dir", paths["unet_dir"])
    add_arg(train_args, "prompt", config.get("experiment", {}).get("prompt", "remove degradation"))

    for key in (
        "seed",
        "resolution",
        "resize_scale",
        "disable_augment",
        "batch_size",
        "num_workers",
        "epochs",
        "max_train_steps",
        "learning_rate",
        "min_learning_rate",
        "lr_warmup_steps",
        "weight_decay",
        "gradient_accumulation_steps",
        "mixed_precision",
        "report_to",
        "logging_dir",
        "checkpointing_steps",
        "log_interval",
        "validation_steps",
        "validation_example_index",
        "checkpoints_total_limit",
        "l1_weight",
        "lpips_weight",
        "grad_weight",
        "nafnet_width",
        "nafnet_middle_blk_num",
        "nafnet_enc_blk_nums",
        "nafnet_dec_blk_nums",
        "beta",
    ):
        add_arg(train_args, key, stage2.get(key))

    if validation_enabled:
        validation_arg_map = {
            "batch_size": "validation_batch_size",
            "num_workers": "validation_num_workers",
            "resolution_mode": "validation_resolution_mode",
            "resize": "validation_resize",
            "num_images": "validation_num_images",
        }
        for config_key, arg_name in validation_arg_map.items():
            add_arg(train_args, arg_name, validation.get(config_key))

    if parsed.dry_run:
        print("python train_refine_cosine.py " + " ".join(map(str, train_args)))
        return

    if get_local_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot = dict(config)
        config_snapshot["paths"] = dict(config["paths"])
        config_snapshot["paths"]["output_dir"] = str(output_dir)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_snapshot, f, sort_keys=False, allow_unicode=True)
        print(f"Resolved output_dir: {output_dir}")
        print(f"Saved config snapshot to {output_dir / 'config.yaml'}")

    from train_refine_cosine import main as train_main, parse_args as parse_train_args

    train_main(parse_train_args([str(arg) for arg in train_args]))


if __name__ == "__main__":
    main()
