"""CSV/directory helpers."""

import os
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_output_dir(output_root: str, video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(output_root, stem)
    os.makedirs(out, exist_ok=True)
    return out


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s (%d rows)", path, len(df))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Saved JSON: %s", path)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    logger.info("Saved NPY: %s shape=%s", path, arr.shape)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def cache_exists(output_dir: str, filenames: list) -> bool:
    """Check if all expected output files already exist."""
    for fn in filenames:
        if not os.path.exists(os.path.join(output_dir, fn)):
            return False
    return True


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)
