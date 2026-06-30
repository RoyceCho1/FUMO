from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def normalize_dirs(paths: Iterable[str | Path] | str | Path | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def normalize_map_array(data: np.ndarray) -> np.ndarray:
    if data.ndim > 2:
        data = data.squeeze()
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data.astype(np.float32), 0.0, 1.0)


def load_map_array(path: str | Path) -> np.ndarray:
    return normalize_map_array(np.load(path))


def load_map_pil(path: str | Path) -> Image.Image:
    data = load_map_array(path)
    return Image.fromarray((data * 255.0).astype(np.uint8), mode="L")


def load_map_tensor(path: str | Path) -> torch.Tensor:
    return TF.to_tensor(load_map_pil(path))


def resolve_m_local_path(
    item: dict,
    conditioning_image_path: str | Path,
    m_local_dirs: Iterable[str | Path] | str | Path | None = None,
    m_local_column: str = "m_local",
    missing_policy: str = "error",
) -> Path | None:
    if missing_policy not in {"error", "zero"}:
        raise ValueError(f"Unsupported m_local_missing_policy: {missing_policy}")

    explicit = item.get(m_local_column) if item is not None else None
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path
        if missing_policy == "error":
            raise FileNotFoundError(f"Missing M_local path from `{m_local_column}`: {explicit_path}")
        return None

    stem = Path(conditioning_image_path).stem
    for root in normalize_dirs(m_local_dirs):
        candidate = root / f"{stem}.npy"
        if candidate.exists():
            return candidate

    if missing_policy == "error":
        searched = ", ".join(str(path) for path in normalize_dirs(m_local_dirs)) or "<none>"
        raise FileNotFoundError(f"Could not resolve M_local for `{conditioning_image_path}` in: {searched}")
    return None
